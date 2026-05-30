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

        let promptTokenIDs = try tokenizer.encode(prompt)

        switch bundle.graphInterface {
        case .logitsAndLayeredKV(let layerCount, _, _):
            return try runLogitsAndLayeredKV(promptTokenIDs: promptTokenIDs, layerCount: layerCount, topK: topK)
        case .statefulStepKV:
            guard #available(macOS 15.0, iOS 18.0, watchOS 11.0, tvOS 18.0, visionOS 2.0, *) else {
                throw InferenceRuntimeError.unavailableRuntime(
                    reason: "Core ML stateful step KV diagnostics require macOS 15, iOS 18, watchOS 11, tvOS 18, or visionOS 2."
                )
            }
            return try runStatefulStepKV(promptTokenIDs: promptTokenIDs, topK: topK)
        case .tokenAndSingleKV, .statefulKV:
            throw InferenceRuntimeError.invalidInput(message: "Diagnostics require logitsAndLayeredKV or statefulStepKV graphs.")
        }
    }

    private func runLogitsAndLayeredKV(
        promptTokenIDs: [Int32],
        layerCount: Int,
        topK: Int
    ) throws -> CoreMLPrefillDecodeDiagnosticReport {
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

    @available(macOS 15.0, iOS 18.0, watchOS 11.0, tvOS 18.0, visionOS 2.0, *)
    private func runStatefulStepKV(
        promptTokenIDs: [Int32],
        topK: Int
    ) throws -> CoreMLPrefillDecodeDiagnosticReport {
        guard !promptTokenIDs.isEmpty else {
            throw InferenceRuntimeError.invalidInput(message: "Prompt produced no tokens for stateful step KV diagnostics.")
        }
        guard bundle.prefillModelURL == bundle.decodeModelURL else {
            throw InferenceRuntimeError.invalidInput(
                message: "Stateful step KV diagnostics require prefill and decode model URLs to reference the same shared model."
            )
        }

        let promptWindow = Array(promptTokenIDs.suffix(bundle.maxPromptTokens))
        var inputState = try CoreMLMiniCPMInputState(
            tokenIDs: [],
            capacity: bundle.maxPromptTokens
        )
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        let model = try diagnosticLoadModel(at: bundle.prefillModelURL, configuration: configuration)
        let state = model.makeState()
        var lastPromptLogits: MLMultiArray?

        for promptTokenID in promptWindow {
            let stepInput = try CoreMLDictionaryFeatureProvider(features: [
                bundle.prefillInputName: MLFeatureValue(multiArray: try tokenIDArray(promptTokenID)),
                bundle.prefillPositionInputName: MLFeatureValue(multiArray: inputState.decodePositionID),
                bundle.prefillCausalMaskInputName: MLFeatureValue(multiArray: inputState.statefulStepCausalMask)
            ])
            let stepOutput = try diagnosticStatefulPrediction(model: model, input: stepInput, state: state)
            lastPromptLogits = try multiArrayOutput(from: stepOutput, name: bundle.prefillLogitsOutputName)
            inputState.appendGeneratedToken()
        }

        guard let prefillLogits = lastPromptLogits else {
            throw InferenceRuntimeError.predictionFailed(message: "Stateful step KV diagnostics did not produce prompt logits.")
        }

        let prefillTopK = topTokenLogits(from: prefillLogits, limit: topK)
        guard let prefillTokenID = prefillTopK.first?.tokenID else {
            throw InferenceRuntimeError.predictionFailed(message: "Prefill logits did not contain any tokens.")
        }

        let decodeInput = try CoreMLDictionaryFeatureProvider(features: [
            bundle.decodeTokenInputName: MLFeatureValue(multiArray: try tokenIDArray(prefillTokenID)),
            bundle.decodePositionInputName: MLFeatureValue(multiArray: inputState.decodePositionID),
            bundle.decodeCausalMaskInputName: MLFeatureValue(multiArray: inputState.statefulStepCausalMask)
        ])
        let decodeOutput = try diagnosticStatefulPrediction(model: model, input: decodeInput, state: state)
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

@available(macOS 15.0, iOS 18.0, watchOS 11.0, tvOS 18.0, visionOS 2.0, *)
private func diagnosticStatefulPrediction(model: MLModel, input: MLFeatureProvider, state: MLState) throws -> MLFeatureProvider {
    do {
        return try model.prediction(from: input, using: state)
    } catch let error as InferenceRuntimeError {
        throw error
    } catch {
        throw InferenceRuntimeError.predictionFailed(message: error.localizedDescription)
    }
}
#endif
