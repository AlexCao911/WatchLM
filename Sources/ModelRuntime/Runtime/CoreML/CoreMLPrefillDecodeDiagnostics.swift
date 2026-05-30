#if canImport(CoreML)
import CoreML
import Foundation

public struct CoreMLPrefillDecodeDiagnosticReport: Codable, Equatable, Sendable {
    public var promptTokenIDs: [Int32]
    public var prefillTopK: [TokenLogit]
    public var decodeTopK: [TokenLogit]

    public init(
        promptTokenIDs: [Int32],
        prefillTopK: [TokenLogit],
        decodeTopK: [TokenLogit]
    ) {
        self.promptTokenIDs = promptTokenIDs
        self.prefillTopK = prefillTopK
        self.decodeTopK = decodeTopK
    }

    public var prefillTokenID: Int32? {
        prefillTopK.first?.tokenID
    }

    public var firstDecodeTokenID: Int32? {
        decodeTopK.first?.tokenID
    }
}

public struct CoreMLPrefillDecodeDiagnostics {
    private let bundle: CoreMLPrefillDecodeBundle
    private let tokenizer: any TextTokenizer

    public init(bundle: CoreMLPrefillDecodeBundle, tokenizer: any TextTokenizer) {
        self.bundle = bundle
        self.tokenizer = tokenizer
    }

    public func run(prompt: String, topK: Int = 10) throws -> CoreMLPrefillDecodeDiagnosticReport {
        guard topK > 0 else {
            throw InferenceRuntimeError.invalidInput(message: "topK must be positive.")
        }

        guard case .logitsAndLayeredKV(let layerCount, _, _) = bundle.graphInterface else {
            throw InferenceRuntimeError.invalidInput(message: "Diagnostics require logitsAndLayeredKV graphs.")
        }

        let promptTokenIDs = try tokenizer.encode(prompt)
        let inputState = try CoreMLMiniCPMInputState(
            tokenIDs: promptTokenIDs,
            capacity: bundle.maxPromptTokens
        )
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        let prefillModel = try diagnosticLoadModel(at: bundle.prefillModelURL, configuration: configuration)
        let decodeModel = try diagnosticLoadModel(at: bundle.decodeModelURL, configuration: configuration)
        let prefillInput = try CoreMLDictionaryFeatureProvider(features: [
            bundle.prefillInputName: MLFeatureValue(multiArray: inputState.inputIDs),
            bundle.prefillPositionInputName: MLFeatureValue(multiArray: inputState.positionIDs),
            bundle.prefillCausalMaskInputName: MLFeatureValue(multiArray: inputState.causalMask)
        ])
        let prefillOutput = try prefillModel.prediction(from: prefillInput)
        let prefillLogits = try multiArrayOutput(from: prefillOutput, name: bundle.prefillLogitsOutputName)
        let prefillTopK = topTokenLogits(from: prefillLogits, limit: topK)
        guard let prefillTokenID = prefillTopK.first?.tokenID else {
            throw InferenceRuntimeError.predictionFailed(message: "Prefill logits did not contain any tokens.")
        }

        let kvCache = try CoreMLKVCacheStore(
            prefillOutput: prefillOutput,
            layerCount: layerCount,
            validTokenCount: inputState.realTokenCount,
            updateStrategy: bundle.kvCacheUpdateStrategy,
            keyOutputName: bundle.prefillKeyOutputName(forLayer:),
            valueOutputName: bundle.prefillValueOutputName(forLayer:)
        )
        var decodeFeatures: [String: MLFeatureValue] = [
            bundle.decodeTokenInputName: MLFeatureValue(multiArray: try tokenIDArray(prefillTokenID)),
            bundle.decodePositionInputName: MLFeatureValue(multiArray: inputState.decodePositionID),
            bundle.decodeCausalMaskInputName: MLFeatureValue(multiArray: inputState.decodeCausalMask)
        ]
        for layer in 0..<layerCount {
            decodeFeatures[bundle.decodePastKeyInputName(forLayer: layer)] = MLFeatureValue(multiArray: kvCache.key(forLayer: layer))
            decodeFeatures[bundle.decodePastValueInputName(forLayer: layer)] = MLFeatureValue(multiArray: kvCache.value(forLayer: layer))
        }

        let decodeOutput = try decodeModel.prediction(from: try CoreMLDictionaryFeatureProvider(features: decodeFeatures))
        let decodeLogits = try multiArrayOutput(from: decodeOutput, name: bundle.decodeLogitsOutputName)
        return CoreMLPrefillDecodeDiagnosticReport(
            promptTokenIDs: promptTokenIDs,
            prefillTopK: prefillTopK,
            decodeTopK: topTokenLogits(from: decodeLogits, limit: topK)
        )
    }
}

private func topTokenLogits(from logits: MLMultiArray, limit: Int) -> [TokenLogit] {
    (0..<logits.count)
        .map { index in
            TokenLogit(tokenID: Int32(index), logit: logits[index].doubleValue)
        }
        .sorted(by: isPreferredToken)
        .prefix(limit)
        .map { $0 }
}

private func diagnosticLoadModel(at url: URL, configuration: MLModelConfiguration) throws -> MLModel {
    #if os(macOS)
    if url.pathExtension == "mlpackage" || url.pathExtension == "mlmodel" {
        let compiledURL = try MLModel.compileModel(at: url)
        return try MLModel(contentsOf: compiledURL, configuration: configuration)
    }
    #endif

    return try MLModel(contentsOf: url, configuration: configuration)
}
#endif
