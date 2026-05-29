public struct InferenceRequest: Codable, Equatable, Sendable {
    public var prompt: String
    public var maxNewTokens: Int

    public init(prompt: String, maxNewTokens: Int) {
        self.prompt = prompt
        self.maxNewTokens = maxNewTokens
    }
}

public enum InferenceTerminationReason: String, Codable, Equatable, Sendable {
    case maxTokens
    case endOfSequence
    case sourceExhausted
}

public struct InferenceResult: Codable, Equatable, Sendable {
    public var tokens: [String]
    public var generatedTokenIDs: [Int32]
    public var timing: RuntimeTiming
    public var metrics: InferenceMetrics
    public var terminationReason: InferenceTerminationReason

    public init(
        tokens: [String],
        generatedTokenIDs: [Int32] = [],
        timing: RuntimeTiming,
        metrics: InferenceMetrics = InferenceMetrics(),
        terminationReason: InferenceTerminationReason = .maxTokens
    ) {
        self.tokens = tokens
        self.generatedTokenIDs = generatedTokenIDs
        self.timing = timing
        self.metrics = metrics
        self.terminationReason = terminationReason
    }

    public var text: String {
        tokens.joined()
    }
}

public struct InferenceToken: Codable, Equatable, Sendable {
    public var index: Int
    public var tokenID: Int32?
    public var text: String
    public var isFirstToken: Bool

    public init(index: Int, tokenID: Int32?, text: String, isFirstToken: Bool) {
        self.index = index
        self.tokenID = tokenID
        self.text = text
        self.isFirstToken = isFirstToken
    }
}

public enum InferenceStreamEvent: Codable, Equatable, Sendable {
    case token(InferenceToken)
    case completed(InferenceResult)
}

public enum InferenceRuntimeError: Error, Codable, Equatable, Sendable {
    case modelAssetMissing
    case cancelled(partialTokens: [String])
    case invalidInput(message: String)
    case predictionFailed(message: String)
    case unavailableRuntime(reason: String)

    public var userMessage: String {
        switch self {
        case .modelAssetMissing:
            "Model asset is not installed."
        case .cancelled:
            "Generation was cancelled."
        case .invalidInput(let message):
            "Invalid input: \(message)"
        case .predictionFailed(let message):
            "Prediction failed: \(message)"
        case .unavailableRuntime(let reason):
            "Runtime unavailable: \(reason)"
        }
    }
}

public protocol InferenceRuntime: Sendable {
    func load() async throws -> RuntimeTiming

    func generate(
        request: InferenceRequest,
        shouldCancel: @Sendable () -> Bool
    ) async throws -> InferenceResult
}

public protocol StreamingInferenceRuntime: InferenceRuntime {
    func stream(
        request: InferenceRequest,
        shouldCancel: @escaping @Sendable () -> Bool
    ) -> AsyncThrowingStream<InferenceStreamEvent, Error>
}
