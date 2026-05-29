import Foundation

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

public struct LogitsProcessor: Codable, Equatable, Sendable {
    public var temperature: Double
    public var topK: Int?
    public var topP: Double?
    public var repetitionPenalty: Double

    public init(
        temperature: Double = 1.0,
        topK: Int? = nil,
        topP: Double? = nil,
        repetitionPenalty: Double = 1.0
    ) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
    }

    public func process(
        logits: [TokenLogit],
        generatedTokenIDs: [Int32] = []
    ) throws -> [TokenLogit] {
        try validate()

        let repeatedTokenIDs = Set(generatedTokenIDs)
        let mapped = logits.map { tokenLogit in
            var logit = tokenLogit.logit
            if repetitionPenalty != 1.0, repeatedTokenIDs.contains(tokenLogit.tokenID) {
                logit = logit >= 0 ? logit / repetitionPenalty : logit * repetitionPenalty
            }
            return TokenLogit(tokenID: tokenLogit.tokenID, logit: logit / temperature)
        }

        guard topK != nil || topP != nil else {
            return mapped
        }

        var processed = mapped.sorted(by: isPreferredToken)

        if let topK {
            processed = Array(processed.prefix(max(0, topK)))
        }

        if let topP, topP < 1.0 {
            processed = nucleusFiltered(processed, threshold: topP)
        }

        return processed
    }

    private func validate() throws {
        guard temperature > 0 else {
            throw InferenceRuntimeError.invalidInput(message: "temperature must be greater than zero.")
        }

        if let topK, topK < 0 {
            throw InferenceRuntimeError.invalidInput(message: "topK must be greater than or equal to zero.")
        }

        if let topP, topP <= 0 || topP > 1 {
            throw InferenceRuntimeError.invalidInput(message: "topP must be in the range (0, 1].")
        }

        guard repetitionPenalty > 0 else {
            throw InferenceRuntimeError.invalidInput(message: "repetitionPenalty must be greater than zero.")
        }
    }

    private func nucleusFiltered(_ logits: [TokenLogit], threshold: Double) -> [TokenLogit] {
        guard !logits.isEmpty else {
            return []
        }

        let maxLogit = logits.map(\.logit).max() ?? 0
        let weights = logits.map { exp($0.logit - maxLogit) }
        let totalWeight = weights.reduce(0, +)
        guard totalWeight > 0, totalWeight.isFinite else {
            return [logits[0]]
        }

        var cumulativeProbability = 0.0
        var selected: [TokenLogit] = []
        for (tokenLogit, weight) in zip(logits, weights) {
            selected.append(tokenLogit)
            cumulativeProbability += weight / totalWeight
            if cumulativeProbability >= threshold {
                break
            }
        }

        return selected
    }
}

public struct GreedyTokenSampler: TokenSampler {
    public init() {}

    public func selectToken(from logits: [TokenLogit]) throws -> Int32 {
        guard var selected = logits.first else {
            throw InferenceRuntimeError.predictionFailed(message: "No logits available for sampling.")
        }

        for tokenLogit in logits.dropFirst() where isPreferredToken(tokenLogit, selected) {
            selected = tokenLogit
        }

        return selected.tokenID
    }
}

public struct SeededRandomNumberGenerator: RandomNumberGenerator, Sendable {
    public private(set) var state: UInt64

    public init(seed: UInt64) {
        state = seed
    }

    public mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }

    public mutating func nextUnitDouble() -> Double {
        Double(next() >> 11) / 9_007_199_254_740_992.0
    }
}

public final class SeededTokenSampler: TokenSampler, @unchecked Sendable {
    private let lock = NSLock()
    private var generator: SeededRandomNumberGenerator

    public init(seed: UInt64) {
        generator = SeededRandomNumberGenerator(seed: seed)
    }

    public func selectToken(from logits: [TokenLogit]) throws -> Int32 {
        guard let first = logits.first else {
            throw InferenceRuntimeError.predictionFailed(message: "No logits available for sampling.")
        }

        guard logits.count > 1 else {
            return first.tokenID
        }

        if let tokenID = try sampleFiniteWeights(from: logits) {
            return tokenID
        }

        return try GreedyTokenSampler().selectToken(from: logits)
    }

    private func sampleFiniteWeights(from logits: [TokenLogit]) throws -> Int32? {
        guard let maxLogit = logits.map(\.logit).max(), maxLogit.isFinite else {
            return nil
        }

        let weighted = logits.compactMap { tokenLogit -> (tokenID: Int32, weight: Double)? in
            let weight = exp(tokenLogit.logit - maxLogit)
            guard weight > 0, weight.isFinite else {
                return nil
            }
            return (tokenID: tokenLogit.tokenID, weight: weight)
        }
        let totalWeight = weighted.reduce(0.0) { $0 + $1.weight }
        guard totalWeight > 0, totalWeight.isFinite else {
            return nil
        }

        let draw = lock.withLock {
            generator.nextUnitDouble()
        } * totalWeight
        var cumulativeWeight = 0.0
        for candidate in weighted {
            cumulativeWeight += candidate.weight
            if draw < cumulativeWeight {
                return candidate.tokenID
            }
        }

        return weighted.last?.tokenID
    }
}

public enum TokenSamplingStrategy: Codable, Equatable, Sendable {
    case greedy
    case seeded(seed: UInt64)

    public func makeSampler() -> any TokenSampler {
        switch self {
        case .greedy:
            GreedyTokenSampler()
        case .seeded(let seed):
            SeededTokenSampler(seed: seed)
        }
    }
}

func isPreferredToken(_ lhs: TokenLogit, _ rhs: TokenLogit) -> Bool {
    if lhs.logit == rhs.logit {
        return lhs.tokenID < rhs.tokenID
    }
    return lhs.logit > rhs.logit
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
