public struct RuntimeTiming: Codable, Equatable, Sendable {
    public var loadMs: Double
    public var prefillMs: Double
    public var firstTokenMs: Double
    public var decodeStepMs: [Double]

    public init(
        loadMs: Double = 0,
        prefillMs: Double = 0,
        firstTokenMs: Double = 0,
        decodeStepMs: [Double] = []
    ) {
        self.loadMs = loadMs
        self.prefillMs = prefillMs
        self.firstTokenMs = firstTokenMs
        self.decodeStepMs = decodeStepMs
    }

    public var totalMs: Double {
        loadMs + prefillMs + firstTokenMs + decodeStepMs.reduce(0, +)
    }

    public var decodeTokensPerSecond: Double {
        let decodeMs = decodeStepMs.reduce(0, +)
        guard decodeMs > 0 else {
            return 0
        }

        let tokensPerSecond = Double(decodeStepMs.count) / decodeMs * 1000
        return (tokensPerSecond * 100).rounded() / 100
    }
}
