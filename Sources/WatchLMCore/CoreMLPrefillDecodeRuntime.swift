#if canImport(CoreML)
import CoreML
import Foundation

public struct CoreMLPrefillDecodeBundle: Sendable {
    public var prefillModelURL: URL
    public var decodeModelURL: URL
    public var maxPromptTokens: Int
    public var prefillInputName: String
    public var prefillNextTokenOutputName: String
    public var prefillKVCacheOutputName: String
    public var decodeTokenInputName: String
    public var decodeKVCacheInputName: String
    public var decodeNextTokenOutputName: String
    public var decodeKVCacheOutputName: String

    public init(
        prefillModelURL: URL,
        decodeModelURL: URL,
        maxPromptTokens: Int,
        prefillInputName: String = "input_ids",
        prefillNextTokenOutputName: String = "next_token",
        prefillKVCacheOutputName: String = "kv_cache",
        decodeTokenInputName: String = "token",
        decodeKVCacheInputName: String = "kv_cache",
        decodeNextTokenOutputName: String = "next_token",
        decodeKVCacheOutputName: String = "updated_kv_cache"
    ) {
        self.prefillModelURL = prefillModelURL
        self.decodeModelURL = decodeModelURL
        self.maxPromptTokens = maxPromptTokens
        self.prefillInputName = prefillInputName
        self.prefillNextTokenOutputName = prefillNextTokenOutputName
        self.prefillKVCacheOutputName = prefillKVCacheOutputName
        self.decodeTokenInputName = decodeTokenInputName
        self.decodeKVCacheInputName = decodeKVCacheInputName
        self.decodeNextTokenOutputName = decodeNextTokenOutputName
        self.decodeKVCacheOutputName = decodeKVCacheOutputName
    }
}

public final class CoreMLPrefillDecodeRuntime: InferenceRuntime, @unchecked Sendable {
    private let bundle: CoreMLPrefillDecodeBundle
    private let tokenizer: any TextTokenizer
    private let lock = NSLock()
    private var loadedModels: LoadedPrefillDecodeModels?

    public init(bundle: CoreMLPrefillDecodeBundle, tokenizer: any TextTokenizer) {
        self.bundle = bundle
        self.tokenizer = tokenizer
    }

    public func load() async throws -> RuntimeTiming {
        let started = Date()
        let models = try loadModels()
        lock.withLock {
            loadedModels = models
        }

        return RuntimeTiming(loadMs: coreMLElapsedMilliseconds(since: started))
    }

    public func generate(
        request: InferenceRequest,
        shouldCancel: @Sendable () -> Bool
    ) async throws -> InferenceResult {
        guard request.maxNewTokens > 0 else {
            return InferenceResult(tokens: [], timing: RuntimeTiming())
        }

        if shouldCancel() {
            throw InferenceRuntimeError.cancelled(partialTokens: [])
        }

        let promptTokens = try tokenizer.encode(request.prompt)
        guard !promptTokens.isEmpty else {
            throw InferenceRuntimeError.invalidInput(message: "Prompt produced no tokens.")
        }

        let models = try currentModels()
        var emittedTokenIDs: [Int32] = []
        var emittedText: [String] = []
        var decodeStepMs: [Double] = []

        let prefillStarted = Date()
        let prefillInput = try CoreMLDictionaryFeatureProvider(features: [
            bundle.prefillInputName: MLFeatureValue(
                multiArray: try paddedPromptArray(
                    tokenIDs: promptTokens,
                    capacity: bundle.maxPromptTokens
                )
            )
        ])
        let prefillOutput = try await prediction(model: models.prefill, input: prefillInput)
        let prefillMs = coreMLElapsedMilliseconds(since: prefillStarted)
        var nextTokenID = try tokenIDOutput(
            from: prefillOutput,
            name: bundle.prefillNextTokenOutputName
        )
        var kvCache = try multiArrayOutput(
            from: prefillOutput,
            name: bundle.prefillKVCacheOutputName
        )

        if !tokenizer.endOfSequenceTokenIDs.contains(nextTokenID) {
            emittedTokenIDs.append(nextTokenID)
            emittedText.append(try tokenizer.decode(tokenIDs: [nextTokenID]))
        }

        while emittedTokenIDs.count < request.maxNewTokens {
            if shouldCancel() {
                throw InferenceRuntimeError.cancelled(partialTokens: emittedText)
            }

            let decodeStarted = Date()
            let decodeInput = try CoreMLDictionaryFeatureProvider(features: [
                bundle.decodeTokenInputName: MLFeatureValue(
                    multiArray: try scalarArray(Double(nextTokenID))
                ),
                bundle.decodeKVCacheInputName: MLFeatureValue(multiArray: kvCache)
            ])
            let decodeOutput = try await prediction(model: models.decode, input: decodeInput)
            decodeStepMs.append(coreMLElapsedMilliseconds(since: decodeStarted))
            nextTokenID = try tokenIDOutput(
                from: decodeOutput,
                name: bundle.decodeNextTokenOutputName
            )
            kvCache = try multiArrayOutput(
                from: decodeOutput,
                name: bundle.decodeKVCacheOutputName
            )

            if tokenizer.endOfSequenceTokenIDs.contains(nextTokenID) {
                break
            }

            emittedTokenIDs.append(nextTokenID)
            emittedText.append(try tokenizer.decode(tokenIDs: [nextTokenID]))
        }

        return InferenceResult(
            tokens: emittedText,
            timing: RuntimeTiming(
                prefillMs: prefillMs,
                firstTokenMs: prefillMs,
                decodeStepMs: decodeStepMs
            )
        )
    }

    private func currentModels() throws -> LoadedPrefillDecodeModels {
        if let models = lock.withLock({ loadedModels }) {
            return models
        }

        let models = try loadModels()
        lock.withLock {
            loadedModels = models
        }
        return models
    }

    private func loadModels() throws -> LoadedPrefillDecodeModels {
        do {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .all
            return LoadedPrefillDecodeModels(
                prefill: try MLModel(contentsOf: bundle.prefillModelURL, configuration: configuration),
                decode: try MLModel(contentsOf: bundle.decodeModelURL, configuration: configuration)
            )
        } catch {
            throw InferenceRuntimeError.unavailableRuntime(reason: "Core ML prefill/decode load failed: \(error.localizedDescription)")
        }
    }
}

private struct LoadedPrefillDecodeModels {
    var prefill: MLModel
    var decode: MLModel
}

private final class CoreMLDictionaryFeatureProvider: MLFeatureProvider {
    private let features: [String: MLFeatureValue]

    var featureNames: Set<String> {
        Set(features.keys)
    }

    init(features: [String: MLFeatureValue]) throws {
        guard !features.isEmpty else {
            throw InferenceRuntimeError.invalidInput(message: "Core ML input features cannot be empty.")
        }

        self.features = features
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        features[featureName]
    }
}

private func prediction(model: MLModel, input: MLFeatureProvider) async throws -> MLFeatureProvider {
    do {
        #if os(watchOS)
        return try await model.prediction(from: input)
        #else
        return try model.prediction(from: input)
        #endif
    } catch let error as InferenceRuntimeError {
        throw error
    } catch {
        throw InferenceRuntimeError.predictionFailed(message: error.localizedDescription)
    }
}

private func paddedPromptArray(tokenIDs: [Int32], capacity: Int) throws -> MLMultiArray {
    guard capacity > 0 else {
        throw InferenceRuntimeError.invalidInput(message: "maxPromptTokens must be positive.")
    }

    let array = try MLMultiArray(shape: [NSNumber(value: capacity)], dataType: .double)
    let suffix = tokenIDs.suffix(capacity)
    for index in 0..<capacity {
        let tokenIndex = suffix.index(suffix.startIndex, offsetBy: index, limitedBy: suffix.index(before: suffix.endIndex))
        let tokenID = tokenIndex.map { suffix[$0] } ?? 0
        array[index] = NSNumber(value: Double(tokenID))
    }
    return array
}

private func scalarArray(_ value: Double) throws -> MLMultiArray {
    let array = try MLMultiArray(shape: [1], dataType: .double)
    array[0] = NSNumber(value: value)
    return array
}

private func tokenIDOutput(from output: MLFeatureProvider, name: String) throws -> Int32 {
    let value = try scalarCoreMLOutput(from: output, name: name)
    guard value.isFinite, value >= 0, value <= Double(Int32.max) else {
        throw InferenceRuntimeError.predictionFailed(message: "Output token \(name) is outside Int32 range.")
    }

    return Int32(value.rounded())
}

private func multiArrayOutput(from output: MLFeatureProvider, name: String) throws -> MLMultiArray {
    guard let value = output.featureValue(for: name) else {
        throw InferenceRuntimeError.predictionFailed(message: "Missing output feature \(name).")
    }

    guard let multiArray = value.multiArrayValue else {
        throw InferenceRuntimeError.predictionFailed(message: "Output feature \(name) is not an MLMultiArray.")
    }

    return multiArray
}

private func scalarCoreMLOutput(from output: MLFeatureProvider, name: String) throws -> Double {
    guard let value = output.featureValue(for: name) else {
        throw InferenceRuntimeError.predictionFailed(message: "Missing output feature \(name).")
    }

    if let multiArray = value.multiArrayValue {
        guard multiArray.count > 0 else {
            throw InferenceRuntimeError.predictionFailed(message: "Output feature \(name) is empty.")
        }
        return multiArray[0].doubleValue
    }

    if value.type == .double {
        return value.doubleValue
    }

    if value.type == .int64 {
        return Double(value.int64Value)
    }

    throw InferenceRuntimeError.predictionFailed(message: "Output feature \(name) is not numeric.")
}

private func coreMLElapsedMilliseconds(since started: Date) -> Double {
    let elapsed = Date().timeIntervalSince(started) * 1000
    return (elapsed * 1000).rounded() / 1000
}
#endif
