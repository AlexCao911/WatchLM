public struct InferenceRequest: Codable, Equatable, Sendable {
    public var prompt: String
    public var maxNewTokens: Int

    public init(prompt: String, maxNewTokens: Int) {
        self.prompt = prompt
        self.maxNewTokens = maxNewTokens
    }
}

public struct InferenceResult: Codable, Equatable, Sendable {
    public var tokens: [String]
    public var timing: RuntimeTiming

    public init(tokens: [String], timing: RuntimeTiming) {
        self.tokens = tokens
        self.timing = timing
    }

    public var text: String {
        tokens.joined()
    }
}

public enum InferenceRuntimeError: Error, Codable, Equatable, Sendable {
    case modelAssetMissing
    case cancelled(partialTokens: [String])
    case unavailableRuntime(reason: String)

    public var userMessage: String {
        switch self {
        case .modelAssetMissing:
            "Model asset is not installed."
        case .cancelled:
            "Generation was cancelled."
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
