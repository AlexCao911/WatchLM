public struct MockStreamingRuntime: InferenceRuntime {
    private let tokens: [String]
    private let loadMs: Double
    private let prefillMs: Double
    private let firstTokenMs: Double
    private let decodeStepMs: [Double]
    private let failure: InferenceRuntimeError?

    public init(
        tokens: [String],
        loadMs: Double = 0,
        prefillMs: Double = 0,
        firstTokenMs: Double = 0,
        decodeStepMs: [Double] = [],
        failure: InferenceRuntimeError? = nil
    ) {
        self.tokens = tokens
        self.loadMs = loadMs
        self.prefillMs = prefillMs
        self.firstTokenMs = firstTokenMs
        self.decodeStepMs = decodeStepMs
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

        return InferenceResult(
            tokens: emitted,
            timing: RuntimeTiming(
                prefillMs: prefillMs,
                firstTokenMs: firstTokenMs,
                decodeStepMs: timingSteps(for: emitted.count)
            )
        )
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
}
