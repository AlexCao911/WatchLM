public enum InferenceSessionState: Codable, Equatable, Sendable {
    case idle
    case prefill
    case decoding(generatedTokens: Int)
    case cancelled
    case finished(totalTokens: Int)
    case failed(message: String)
    case thermalDegraded

    public var isTerminal: Bool {
        switch self {
        case .cancelled, .finished, .failed, .thermalDegraded:
            true
        case .idle, .prefill, .decoding:
            false
        }
    }
}
