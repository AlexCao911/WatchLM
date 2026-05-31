#if canImport(CoreML)
import CoreML
import Foundation

public final class CoreMLPrefillDecodeRuntime: StreamingInferenceRuntime, @unchecked Sendable {
    private let bundle: CoreMLPrefillDecodeBundle
    private let tokenizer: any TextTokenizer
    private let computeUnits: MLComputeUnits
    private let lock = NSLock()
    private var loadedModels: LoadedPrefillDecodeModels?

    public init(
        bundle: CoreMLPrefillDecodeBundle,
        tokenizer: any TextTokenizer,
        computeUnits: MLComputeUnits = .all
    ) {
        self.bundle = bundle
        self.tokenizer = tokenizer
        self.computeUnits = computeUnits
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
        try await generateInternal(
            request: request,
            shouldCancel: shouldCancel,
            onToken: nil
        )
    }

    public func stream(
        request: InferenceRequest,
        shouldCancel: @escaping @Sendable () -> Bool
    ) -> AsyncThrowingStream<InferenceStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    let result = try await self.generateInternal(
                        request: request,
                        shouldCancel: shouldCancel
                    ) { token in
                        continuation.yield(.token(token))
                    }
                    continuation.yield(.completed(result))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    private func generateInternal(
        request: InferenceRequest,
        shouldCancel: @Sendable () -> Bool,
        onToken: (@Sendable (InferenceToken) -> Void)?
    ) async throws -> InferenceResult {
        guard request.maxNewTokens > 0 else {
            return InferenceResult(tokens: [], timing: RuntimeTiming(), terminationReason: .maxTokens)
        }

        if shouldCancel() {
            throw InferenceRuntimeError.cancelled(partialTokens: [])
        }

        let promptTokens = try tokenizer.encode(request.prompt)
        guard !promptTokens.isEmpty else {
            throw InferenceRuntimeError.invalidInput(message: "Prompt produced no tokens.")
        }

        let models = try currentModels()

        switch bundle.graphInterface {
        case .tokenAndSingleKV:
            return try await generateTokenAndSingleKV(
                request: request,
                promptTokens: promptTokens,
                models: models,
                onToken: onToken,
                shouldCancel: shouldCancel
            )
        case .statefulKV:
            guard #available(macOS 15.0, iOS 18.0, watchOS 11.0, tvOS 18.0, visionOS 2.0, *) else {
                throw InferenceRuntimeError.unavailableRuntime(
                    reason: "Core ML stateful KV requires macOS 15, iOS 18, watchOS 11, tvOS 18, or visionOS 2."
                )
            }
            return try await generateStatefulKV(
                request: request,
                promptTokens: promptTokens,
                models: models,
                onToken: onToken,
                shouldCancel: shouldCancel
            )
        case .statefulStepKV:
            guard #available(macOS 15.0, iOS 18.0, watchOS 11.0, tvOS 18.0, visionOS 2.0, *) else {
                throw InferenceRuntimeError.unavailableRuntime(
                    reason: "Core ML stateful step KV requires macOS 15, iOS 18, watchOS 11, tvOS 18, or visionOS 2."
                )
            }
            return try await generateStatefulStepKV(
                request: request,
                promptTokens: promptTokens,
                models: models,
                onToken: onToken,
                shouldCancel: shouldCancel
            )
        case .logitsAndLayeredKV(let layerCount, _, _):
            return try await generateLogitsAndLayeredKV(
                request: request,
                promptTokens: promptTokens,
                models: models,
                layerCount: layerCount,
                onToken: onToken,
                shouldCancel: shouldCancel
            )
        }
    }

    private func generateTokenAndSingleKV(
        request: InferenceRequest,
        promptTokens: [Int32],
        models: LoadedPrefillDecodeModels,
        onToken: (@Sendable (InferenceToken) -> Void)?,
        shouldCancel: @Sendable () -> Bool
    ) async throws -> InferenceResult {
        var emittedTokenIDs: [Int32] = []
        var emittedText: [String] = []
        var decodeStepMs: [Double] = []
        var terminationReason = InferenceTerminationReason.maxTokens
        let stopCriteria = DecodeStopCriteria(
            maxNewTokens: request.maxNewTokens,
            eosTokenIDs: tokenizer.endOfSequenceTokenIDs
        )

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

        if tokenizer.endOfSequenceTokenIDs.contains(nextTokenID) {
            terminationReason = .endOfSequence
        } else {
            let text = try tokenizer.decode(tokenIDs: [nextTokenID])
            onToken?(InferenceToken(
                index: emittedTokenIDs.count,
                tokenID: nextTokenID,
                text: text,
                isFirstToken: emittedTokenIDs.isEmpty
            ))
            emittedTokenIDs.append(nextTokenID)
            emittedText.append(text)
        }

        while !stopCriteria.shouldStop(generatedTokenIDs: emittedTokenIDs) && terminationReason != .endOfSequence {
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
                terminationReason = .endOfSequence
                break
            }

            let text = try tokenizer.decode(tokenIDs: [nextTokenID])
            onToken?(InferenceToken(
                index: emittedTokenIDs.count,
                tokenID: nextTokenID,
                text: text,
                isFirstToken: emittedTokenIDs.isEmpty
            ))
            emittedTokenIDs.append(nextTokenID)
            emittedText.append(text)
        }

        return InferenceResult(
            tokens: emittedText,
            generatedTokenIDs: emittedTokenIDs,
            timing: RuntimeTiming(
                prefillMs: prefillMs,
                firstTokenMs: prefillMs,
                decodeStepMs: decodeStepMs
            ),
            terminationReason: terminationReason
        )
    }

    @available(macOS 15.0, iOS 18.0, watchOS 11.0, tvOS 18.0, visionOS 2.0, *)
    private func generateStatefulKV(
        request: InferenceRequest,
        promptTokens: [Int32],
        models: LoadedPrefillDecodeModels,
        onToken: (@Sendable (InferenceToken) -> Void)?,
        shouldCancel: @Sendable () -> Bool
    ) async throws -> InferenceResult {
        guard models.prefill === models.decode else {
            throw InferenceRuntimeError.unavailableRuntime(
                reason: "Stateful KV requires prefill and decode to use one shared Core ML model instance."
            )
        }

        var inputState = try CoreMLMiniCPMInputState(
            tokenIDs: promptTokens,
            capacity: bundle.maxPromptTokens,
            reservedGeneratedTokenSlots: max(request.maxNewTokens - 1, 0)
        )
        var emittedTokenIDs: [Int32] = []
        var emittedText: [String] = []
        var decodeStepMs: [Double] = []
        var metrics = InferenceMetrics()
        var terminationReason = InferenceTerminationReason.maxTokens
        let stopCriteria = DecodeStopCriteria(
            maxNewTokens: request.maxNewTokens,
            eosTokenIDs: tokenizer.endOfSequenceTokenIDs
        )
        let logitsSampler = CoreMLLogitsSampler(
            processor: CoreMLLogitsProcessor(
                policy: bundle.logitsProcessor,
                tokenIDUpperBound: tokenizer.decodableTokenIDUpperBound
            ),
            sampler: bundle.samplingStrategy.makeSampler()
        )
        let state = models.prefill.makeState()

        let prefillStarted = Date()
        let prefillInput = try CoreMLDictionaryFeatureProvider(features: [
            bundle.prefillInputName: MLFeatureValue(multiArray: inputState.statefulPrefillInputIDs),
            bundle.prefillPositionInputName: MLFeatureValue(multiArray: inputState.statefulPrefillPositionIDs),
            bundle.prefillCausalMaskInputName: MLFeatureValue(multiArray: inputState.statefulPrefillCausalMask)
        ])
        let prefillOutput = try await statefulPrediction(
            model: models.prefill,
            input: prefillInput,
            state: state
        )
        let prefillMs = coreMLElapsedMilliseconds(since: prefillStarted)
        let prefillLogits = try multiArrayOutput(from: prefillOutput, name: bundle.prefillLogitsOutputName)
        let prefillSamplingStarted = Date()
        var nextTokenID = try logitsSampler.selectToken(from: prefillLogits)
        metrics.prefillSamplingMs = coreMLElapsedMilliseconds(since: prefillSamplingStarted)

        if tokenizer.endOfSequenceTokenIDs.contains(nextTokenID) {
            terminationReason = .endOfSequence
        } else {
            let text = try tokenizer.decode(tokenIDs: [nextTokenID])
            onToken?(InferenceToken(
                index: emittedTokenIDs.count,
                tokenID: nextTokenID,
                text: text,
                isFirstToken: emittedTokenIDs.isEmpty
            ))
            emittedTokenIDs.append(nextTokenID)
            emittedText.append(text)
        }

        while !stopCriteria.shouldStop(generatedTokenIDs: emittedTokenIDs) && terminationReason != .endOfSequence {
            if shouldCancel() {
                throw InferenceRuntimeError.cancelled(partialTokens: emittedText)
            }
            guard inputState.hasStatefulDecodeCapacity else {
                terminationReason = .maxTokens
                break
            }

            let decodeStarted = Date()
            let decodeInput = try CoreMLDictionaryFeatureProvider(features: [
                bundle.decodeTokenInputName: MLFeatureValue(multiArray: try tokenIDArray(nextTokenID)),
                bundle.decodePositionInputName: MLFeatureValue(multiArray: inputState.decodePositionID),
                bundle.decodeCausalMaskInputName: MLFeatureValue(multiArray: inputState.statefulDecodeCausalMask)
            ])
            let decodeOutput = try await statefulPrediction(
                model: models.decode,
                input: decodeInput,
                state: state
            )
            decodeStepMs.append(coreMLElapsedMilliseconds(since: decodeStarted))
            let decodeLogits = try multiArrayOutput(from: decodeOutput, name: bundle.decodeLogitsOutputName)
            let decodeSamplingStarted = Date()
            nextTokenID = try logitsSampler.selectToken(
                from: decodeLogits,
                generatedTokenIDs: emittedTokenIDs
            )
            metrics.decodeSamplingStepMs.append(coreMLElapsedMilliseconds(since: decodeSamplingStarted))
            inputState.appendGeneratedToken()

            if tokenizer.endOfSequenceTokenIDs.contains(nextTokenID) {
                terminationReason = .endOfSequence
                break
            }

            let text = try tokenizer.decode(tokenIDs: [nextTokenID])
            onToken?(InferenceToken(
                index: emittedTokenIDs.count,
                tokenID: nextTokenID,
                text: text,
                isFirstToken: emittedTokenIDs.isEmpty
            ))
            emittedTokenIDs.append(nextTokenID)
            emittedText.append(text)
        }

        return InferenceResult(
            tokens: emittedText,
            generatedTokenIDs: emittedTokenIDs,
            timing: RuntimeTiming(
                prefillMs: prefillMs,
                firstTokenMs: prefillMs,
                decodeStepMs: decodeStepMs
            ),
            metrics: metrics,
            terminationReason: terminationReason
        )
    }

    @available(macOS 15.0, iOS 18.0, watchOS 11.0, tvOS 18.0, visionOS 2.0, *)
    private func generateStatefulStepKV(
        request: InferenceRequest,
        promptTokens: [Int32],
        models: LoadedPrefillDecodeModels,
        onToken: (@Sendable (InferenceToken) -> Void)?,
        shouldCancel: @Sendable () -> Bool
    ) async throws -> InferenceResult {
        guard models.prefill === models.decode else {
            throw InferenceRuntimeError.unavailableRuntime(
                reason: "Stateful step KV requires prefill and decode to use one shared Core ML model instance."
            )
        }

        let promptWindow = Array(promptTokens.suffix(bundle.maxPromptTokens))
        var inputState = try CoreMLMiniCPMInputState(
            tokenIDs: [],
            capacity: bundle.maxPromptTokens
        )
        var emittedTokenIDs: [Int32] = []
        var emittedText: [String] = []
        var decodeStepMs: [Double] = []
        var metrics = InferenceMetrics()
        var terminationReason = InferenceTerminationReason.maxTokens
        let stopCriteria = DecodeStopCriteria(
            maxNewTokens: request.maxNewTokens,
            eosTokenIDs: tokenizer.endOfSequenceTokenIDs
        )
        let logitsSampler = CoreMLLogitsSampler(
            processor: CoreMLLogitsProcessor(
                policy: bundle.logitsProcessor,
                tokenIDUpperBound: tokenizer.decodableTokenIDUpperBound
            ),
            sampler: bundle.samplingStrategy.makeSampler()
        )
        let state = models.prefill.makeState()

        let prefillStarted = Date()
        var lastPromptLogits: MLMultiArray?
        for promptTokenID in promptWindow {
            if shouldCancel() {
                throw InferenceRuntimeError.cancelled(partialTokens: emittedText)
            }

            let stepInput = try CoreMLDictionaryFeatureProvider(features: [
                bundle.prefillInputName: MLFeatureValue(multiArray: try tokenIDArray(promptTokenID)),
                bundle.prefillPositionInputName: MLFeatureValue(multiArray: inputState.decodePositionID),
                bundle.prefillCausalMaskInputName: MLFeatureValue(multiArray: inputState.statefulStepCausalMask)
            ])
            let stepOutput = try await statefulPrediction(
                model: models.prefill,
                input: stepInput,
                state: state
            )
            lastPromptLogits = try multiArrayOutput(from: stepOutput, name: bundle.prefillLogitsOutputName)
            inputState.appendGeneratedToken()
        }

        let prefillMs = coreMLElapsedMilliseconds(since: prefillStarted)
        guard let prefillLogits = lastPromptLogits else {
            throw InferenceRuntimeError.invalidInput(message: "Prompt produced no tokens for stateful step KV.")
        }

        let prefillSamplingStarted = Date()
        var nextTokenID = try logitsSampler.selectToken(from: prefillLogits)
        metrics.prefillSamplingMs = coreMLElapsedMilliseconds(since: prefillSamplingStarted)

        if tokenizer.endOfSequenceTokenIDs.contains(nextTokenID) {
            terminationReason = .endOfSequence
        } else {
            let text = try tokenizer.decode(tokenIDs: [nextTokenID])
            onToken?(InferenceToken(
                index: emittedTokenIDs.count,
                tokenID: nextTokenID,
                text: text,
                isFirstToken: emittedTokenIDs.isEmpty
            ))
            emittedTokenIDs.append(nextTokenID)
            emittedText.append(text)
        }

        while !stopCriteria.shouldStop(generatedTokenIDs: emittedTokenIDs) && terminationReason != .endOfSequence {
            if shouldCancel() {
                throw InferenceRuntimeError.cancelled(partialTokens: emittedText)
            }

            let decodeStarted = Date()
            let decodeInput = try CoreMLDictionaryFeatureProvider(features: [
                bundle.decodeTokenInputName: MLFeatureValue(multiArray: try tokenIDArray(nextTokenID)),
                bundle.decodePositionInputName: MLFeatureValue(multiArray: inputState.decodePositionID),
                bundle.decodeCausalMaskInputName: MLFeatureValue(multiArray: inputState.statefulStepCausalMask)
            ])
            let decodeOutput = try await statefulPrediction(
                model: models.decode,
                input: decodeInput,
                state: state
            )
            decodeStepMs.append(coreMLElapsedMilliseconds(since: decodeStarted))
            let decodeLogits = try multiArrayOutput(from: decodeOutput, name: bundle.decodeLogitsOutputName)
            let decodeSamplingStarted = Date()
            nextTokenID = try logitsSampler.selectToken(
                from: decodeLogits,
                generatedTokenIDs: emittedTokenIDs
            )
            metrics.decodeSamplingStepMs.append(coreMLElapsedMilliseconds(since: decodeSamplingStarted))
            inputState.appendGeneratedToken()

            if tokenizer.endOfSequenceTokenIDs.contains(nextTokenID) {
                terminationReason = .endOfSequence
                break
            }

            let text = try tokenizer.decode(tokenIDs: [nextTokenID])
            onToken?(InferenceToken(
                index: emittedTokenIDs.count,
                tokenID: nextTokenID,
                text: text,
                isFirstToken: emittedTokenIDs.isEmpty
            ))
            emittedTokenIDs.append(nextTokenID)
            emittedText.append(text)
        }

        return InferenceResult(
            tokens: emittedText,
            generatedTokenIDs: emittedTokenIDs,
            timing: RuntimeTiming(
                prefillMs: prefillMs,
                firstTokenMs: prefillMs,
                decodeStepMs: decodeStepMs
            ),
            metrics: metrics,
            terminationReason: terminationReason
        )
    }

    private func generateLogitsAndLayeredKV(
        request: InferenceRequest,
        promptTokens: [Int32],
        models: LoadedPrefillDecodeModels,
        layerCount: Int,
        onToken: (@Sendable (InferenceToken) -> Void)?,
        shouldCancel: @Sendable () -> Bool
    ) async throws -> InferenceResult {
        var inputState = try CoreMLMiniCPMInputState(
            tokenIDs: promptTokens,
            capacity: bundle.maxPromptTokens
        )
        var emittedTokenIDs: [Int32] = []
        var emittedText: [String] = []
        var decodeStepMs: [Double] = []
        var metrics = InferenceMetrics(kvCacheUpdateStrategy: bundle.kvCacheUpdateStrategy)
        var terminationReason = InferenceTerminationReason.maxTokens
        let stopCriteria = DecodeStopCriteria(
            maxNewTokens: request.maxNewTokens,
            eosTokenIDs: tokenizer.endOfSequenceTokenIDs
        )
        let logitsSampler = CoreMLLogitsSampler(
            processor: CoreMLLogitsProcessor(
                policy: bundle.logitsProcessor,
                tokenIDUpperBound: tokenizer.decodableTokenIDUpperBound
            ),
            sampler: bundle.samplingStrategy.makeSampler()
        )

        let prefillStarted = Date()
        let prefillInput = try CoreMLDictionaryFeatureProvider(features: [
            bundle.prefillInputName: MLFeatureValue(multiArray: inputState.inputIDs),
            bundle.prefillPositionInputName: MLFeatureValue(multiArray: inputState.positionIDs),
            bundle.prefillCausalMaskInputName: MLFeatureValue(multiArray: inputState.causalMask)
        ])
        let prefillOutput = try await prediction(model: models.prefill, input: prefillInput)
        let prefillMs = coreMLElapsedMilliseconds(since: prefillStarted)
        let prefillLogits = try multiArrayOutput(from: prefillOutput, name: bundle.prefillLogitsOutputName)
        let prefillSamplingStarted = Date()
        var nextTokenID = try logitsSampler.selectToken(
            from: prefillLogits
        )
        metrics.prefillSamplingMs = coreMLElapsedMilliseconds(since: prefillSamplingStarted)
        var kvCache = try CoreMLKVCacheStore(
            prefillOutput: prefillOutput,
            layerCount: layerCount,
            validTokenCount: inputState.realTokenCount,
            updateStrategy: bundle.kvCacheUpdateStrategy,
            keyOutputName: bundle.prefillKeyOutputName(forLayer:),
            valueOutputName: bundle.prefillValueOutputName(forLayer:)
        )

        if tokenizer.endOfSequenceTokenIDs.contains(nextTokenID) {
            terminationReason = .endOfSequence
        } else {
            let text = try tokenizer.decode(tokenIDs: [nextTokenID])
            onToken?(InferenceToken(
                index: emittedTokenIDs.count,
                tokenID: nextTokenID,
                text: text,
                isFirstToken: emittedTokenIDs.isEmpty
            ))
            emittedTokenIDs.append(nextTokenID)
            emittedText.append(text)
        }

        while !stopCriteria.shouldStop(generatedTokenIDs: emittedTokenIDs) && terminationReason != .endOfSequence {
            if shouldCancel() {
                throw InferenceRuntimeError.cancelled(partialTokens: emittedText)
            }

            let decodeStarted = Date()
            let decodeInput = try layeredDecodeInput(
                tokenID: nextTokenID,
                inputState: inputState,
                kvCache: kvCache,
                layerCount: layerCount
            )
            let decodeOutput = try await prediction(model: models.decode, input: decodeInput)
            decodeStepMs.append(coreMLElapsedMilliseconds(since: decodeStarted))
            let decodeLogits = try multiArrayOutput(from: decodeOutput, name: bundle.decodeLogitsOutputName)
            let decodeSamplingStarted = Date()
            nextTokenID = try logitsSampler.selectToken(
                from: decodeLogits,
                generatedTokenIDs: emittedTokenIDs
            )
            metrics.decodeSamplingStepMs.append(coreMLElapsedMilliseconds(since: decodeSamplingStarted))
            let kvAppendStarted = Date()
            try kvCache.appendDecodeOutputs(
                output: decodeOutput,
                keyOutputName: bundle.decodeNewKeyOutputName(forLayer:),
                valueOutputName: bundle.decodeNewValueOutputName(forLayer:)
            )
            metrics.kvAppendStepMs.append(coreMLElapsedMilliseconds(since: kvAppendStarted))
            metrics.kvAppendWriteIndices.append(kvCache.lastAppendWriteIndex)
            metrics.kvAppendMovedTokenSlots.append(kvCache.lastAppendMovedTokenCount)
            metrics.kvAppendMovedScalarCounts.append(kvCache.lastAppendMovedScalarCount)
            switch bundle.kvCacheUpdateStrategy {
            case .contiguousSliding:
                inputState.appendGeneratedToken()
            case .slotRing:
                try inputState.appendGeneratedToken(atPastKVSlot: kvCache.lastAppendWriteIndex)
            }

            if tokenizer.endOfSequenceTokenIDs.contains(nextTokenID) {
                terminationReason = .endOfSequence
                break
            }

            let text = try tokenizer.decode(tokenIDs: [nextTokenID])
            onToken?(InferenceToken(
                index: emittedTokenIDs.count,
                tokenID: nextTokenID,
                text: text,
                isFirstToken: emittedTokenIDs.isEmpty
            ))
            emittedTokenIDs.append(nextTokenID)
            emittedText.append(text)
        }

        return InferenceResult(
            tokens: emittedText,
            generatedTokenIDs: emittedTokenIDs,
            timing: RuntimeTiming(
                prefillMs: prefillMs,
                firstTokenMs: prefillMs,
                decodeStepMs: decodeStepMs
            ),
            metrics: metrics,
            terminationReason: terminationReason
        )
    }

    private func layeredDecodeInput(
        tokenID: Int32,
        inputState: CoreMLMiniCPMInputState,
        kvCache: CoreMLKVCacheStore,
        layerCount: Int
    ) throws -> MLFeatureProvider {
        var features: [String: MLFeatureValue] = [
            bundle.decodeTokenInputName: MLFeatureValue(multiArray: try tokenIDArray(tokenID)),
            bundle.decodePositionInputName: MLFeatureValue(multiArray: inputState.decodePositionID),
            bundle.decodeCausalMaskInputName: MLFeatureValue(multiArray: inputState.decodeCausalMask)
        ]

        for layer in 0..<layerCount {
            features[bundle.decodePastKeyInputName(forLayer: layer)] = MLFeatureValue(multiArray: kvCache.key(forLayer: layer))
            features[bundle.decodePastValueInputName(forLayer: layer)] = MLFeatureValue(multiArray: kvCache.value(forLayer: layer))
        }

        return try CoreMLDictionaryFeatureProvider(features: features)
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
            configuration.computeUnits = computeUnits
            let prefill = try loadModel(at: bundle.prefillModelURL, configuration: configuration)
            let decode: MLModel
            if bundle.requiresSharedStatefulModel {
                guard bundle.prefillModelURL == bundle.decodeModelURL else {
                    throw InferenceRuntimeError.invalidInput(
                        message: "Stateful KV Core ML graph requires prefill and decode model URLs to reference the same shared model."
                    )
                }
                decode = prefill
            } else {
                decode = try loadModel(at: bundle.decodeModelURL, configuration: configuration)
            }
            let models = LoadedPrefillDecodeModels(
                prefill: prefill,
                decode: decode
            )
            try bundle.validateModelDescriptions(
                prefill: models.prefill.modelDescription,
                decode: models.decode.modelDescription
            )
            return models
        } catch let error as InferenceRuntimeError {
            throw error
        } catch {
            throw InferenceRuntimeError.unavailableRuntime(reason: "Core ML prefill/decode load failed: \(error.localizedDescription)")
        }
    }
}

private struct LoadedPrefillDecodeModels {
    var prefill: MLModel
    var decode: MLModel
}

private func loadModel(at url: URL, configuration: MLModelConfiguration) throws -> MLModel {
    #if os(macOS)
    if url.pathExtension == "mlpackage" || url.pathExtension == "mlmodel" {
        let compiledURL = try MLModel.compileModel(at: url)
        return try MLModel(contentsOf: compiledURL, configuration: configuration)
    }
    #endif

    return try MLModel(contentsOf: url, configuration: configuration)
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

@available(macOS 15.0, iOS 18.0, watchOS 11.0, tvOS 18.0, visionOS 2.0, *)
private func statefulPrediction(model: MLModel, input: MLFeatureProvider, state: MLState) async throws -> MLFeatureProvider {
    do {
        return try await model.prediction(from: input, using: state)
    } catch let error as InferenceRuntimeError {
        throw error
    } catch {
        throw InferenceRuntimeError.predictionFailed(message: error.localizedDescription)
    }
}

private func coreMLElapsedMilliseconds(since started: Date) -> Double {
    let elapsed = Date().timeIntervalSince(started) * 1000
    return (elapsed * 1000).rounded() / 1000
}
#endif
