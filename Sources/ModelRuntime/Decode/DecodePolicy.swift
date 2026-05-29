public struct TokenLogit: Codable, Equatable, Sendable {
    public var tokenID: Int32
    public var logit: Double

    public init(tokenID: Int32, logit: Double) {
        self.tokenID = tokenID
        self.logit = logit
    }
}

public protocol TokenSampler: Sendable {
    func selectToken(from logits: [TokenLogit]) throws -> Int32
}

public struct GreedyTokenSampler: TokenSampler {
    public init() {}

    public func selectToken(from logits: [TokenLogit]) throws -> Int32 {
        guard let selected = logits.sorted(by: isPreferredToken).first else {
            throw InferenceRuntimeError.predictionFailed(message: "No logits available for sampling.")
        }

        return selected.tokenID
    }

    private func isPreferredToken(_ lhs: TokenLogit, _ rhs: TokenLogit) -> Bool {
        if lhs.logit == rhs.logit {
            return lhs.tokenID < rhs.tokenID
        }
        return lhs.logit > rhs.logit
    }
}

public struct DecodeStopCriteria: Codable, Equatable, Sendable {
    public var maxNewTokens: Int
    public var eosTokenIDs: Set<Int32>

    public init(maxNewTokens: Int, eosTokenIDs: Set<Int32>) {
        self.maxNewTokens = maxNewTokens
        self.eosTokenIDs = eosTokenIDs
    }

    public func shouldStop(generatedTokenIDs: [Int32]) -> Bool {
        guard !generatedTokenIDs.isEmpty else {
            return false
        }

        if generatedTokenIDs.count >= maxNewTokens {
            return true
        }

        guard let lastTokenID = generatedTokenIDs.last else {
            return false
        }

        return eosTokenIDs.contains(lastTokenID)
    }
}
