public struct MockStreamingRuntime: StreamingInferenceRuntime {
    private let tokens: [String]
    private let generatedTokenIDs: [Int32]
    private let loadMs: Double
    private let prefillMs: Double
    private let firstTokenMs: Double
    private let decodeStepMs: [Double]
    private let metrics: InferenceMetrics
    private let failure: InferenceRuntimeError?

    public init(
        tokens: [String],
        generatedTokenIDs: [Int32] = [],
        loadMs: Double = 0,
        prefillMs: Double = 0,
        firstTokenMs: Double = 0,
        decodeStepMs: [Double] = [],
        metrics: InferenceMetrics = InferenceMetrics(),
        failure: InferenceRuntimeError? = nil
    ) {
        self.tokens = tokens
        self.generatedTokenIDs = generatedTokenIDs
        self.loadMs = loadMs
        self.prefillMs = prefillMs
        self.firstTokenMs = firstTokenMs
        self.decodeStepMs = decodeStepMs
        self.metrics = metrics
        self.failure = failure
    }

    public func load() async throws -> RuntimeTiming {
        if let failure {
            throw failure
        }

        return RuntimeTiming(loadMs: loadMs)
    }

    public func generate(
        request: InferenceRequest,
        shouldCancel: @Sendable () -> Bool
    ) async throws -> InferenceResult {
        if let failure {
            throw failure
        }

        var emitted: [String] = []
        for token in tokens.prefix(max(0, request.maxNewTokens)) {
            if shouldCancel() {
                throw InferenceRuntimeError.cancelled(partialTokens: emitted)
            }

            emitted.append(token)
        }
        let terminationReason: InferenceTerminationReason =
            emitted.count >= request.maxNewTokens ? .maxTokens : .sourceExhausted

        return InferenceResult(
            tokens: emitted,
            generatedTokenIDs: generatedIDs(for: emitted.count),
            timing: RuntimeTiming(
                prefillMs: prefillMs,
                firstTokenMs: firstTokenMs,
                decodeStepMs: timingSteps(for: emitted.count)
            ),
            metrics: metrics.truncated(to: emitted.count),
            terminationReason: terminationReason
        )
    }

    public func stream(
        request: InferenceRequest,
        shouldCancel: @escaping @Sendable () -> Bool
    ) -> AsyncThrowingStream<InferenceStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    if let failure {
                        throw failure
                    }

                    var emitted: [String] = []
                    for token in tokens.prefix(max(0, request.maxNewTokens)) {
                        if shouldCancel() {
                            throw InferenceRuntimeError.cancelled(partialTokens: emitted)
                        }

                        let event = InferenceToken(
                            index: emitted.count,
                            tokenID: generatedTokenID(at: emitted.count),
                            text: token,
                            isFirstToken: emitted.isEmpty
                        )
                        emitted.append(token)
                        continuation.yield(.token(event))
                    }
                    let terminationReason: InferenceTerminationReason =
                        emitted.count >= request.maxNewTokens ? .maxTokens : .sourceExhausted
                    let result = InferenceResult(
                        tokens: emitted,
                        generatedTokenIDs: generatedIDs(for: emitted.count),
                        timing: RuntimeTiming(
                            prefillMs: prefillMs,
                            firstTokenMs: firstTokenMs,
                            decodeStepMs: timingSteps(for: emitted.count)
                        ),
                        metrics: metrics.truncated(to: emitted.count),
                        terminationReason: terminationReason
                    )
                    continuation.yield(.completed(result))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    private func timingSteps(for tokenCount: Int) -> [Double] {
        guard tokenCount > 0 else {
            return []
        }

        return (0..<tokenCount).map { index in
            if decodeStepMs.indices.contains(index) {
                decodeStepMs[index]
            } else {
                0
            }
        }
    }

    private func generatedTokenID(at index: Int) -> Int32? {
        generatedTokenIDs.indices.contains(index) ? generatedTokenIDs[index] : nil
    }

    private func generatedIDs(for tokenCount: Int) -> [Int32] {
        Array(generatedTokenIDs.prefix(tokenCount))
    }
}

private extension InferenceMetrics {
    func truncated(to stepCount: Int) -> InferenceMetrics {
        InferenceMetrics(
            prefillSamplingMs: prefillSamplingMs,
            decodeSamplingStepMs: Array(decodeSamplingStepMs.prefix(stepCount)),
            kvAppendStepMs: Array(kvAppendStepMs.prefix(stepCount)),
            kvCacheUpdateStrategy: kvCacheUpdateStrategy,
            kvAppendWriteIndices: Array(kvAppendWriteIndices.prefix(stepCount)),
            kvAppendMovedTokenSlots: Array(kvAppendMovedTokenSlots.prefix(stepCount)),
            kvAppendMovedScalarCounts: Array(kvAppendMovedScalarCounts.prefix(stepCount))
        )
    }
}
