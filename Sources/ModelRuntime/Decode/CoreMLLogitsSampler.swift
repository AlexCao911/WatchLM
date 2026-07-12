#if canImport(CoreML)
import CoreML

struct CoreMLLogitsProcessor {
    var policy: LogitsProcessor
    var tokenIDUpperBound: Int32?

    init(topK: Int? = nil, tokenIDUpperBound: Int32? = nil) {
        policy = LogitsProcessor(topK: topK)
        self.tokenIDUpperBound = tokenIDUpperBound
    }

    init(policy: LogitsProcessor, tokenIDUpperBound: Int32? = nil) {
        self.policy = policy
        self.tokenIDUpperBound = tokenIDUpperBound
    }

    func tokenLogits(
        from logits: MLMultiArray,
        generatedTokenIDs: [Int32] = []
    ) throws -> [TokenLogit] {
        guard logits.count > 0 else {
            throw InferenceRuntimeError.predictionFailed(message: "Logits output is empty.")
        }

        let effectiveCount = tokenIDUpperBound.map { max(0, min(logits.count, Int($0))) } ?? logits.count
        let tokenLogits = (0..<effectiveCount).map { tokenID in
            TokenLogit(tokenID: Int32(tokenID), logit: logits[tokenID].doubleValue)
        }
        return try policy.process(logits: tokenLogits, generatedTokenIDs: generatedTokenIDs)
    }
}

struct CoreMLLogitsSampler {
    private let processor: CoreMLLogitsProcessor
    private let sampler: any TokenSampler

    init(
        processor: CoreMLLogitsProcessor = CoreMLLogitsProcessor(),
        sampler: any TokenSampler = GreedyTokenSampler()
    ) {
        self.processor = processor
        self.sampler = sampler
    }

    func selectToken(
        from logits: MLMultiArray,
        generatedTokenIDs: [Int32] = []
    ) throws -> Int32 {
        try sampler.selectToken(from: processor.tokenLogits(from: logits, generatedTokenIDs: generatedTokenIDs))
    }
}

typealias CoreMLLogitsGreedySampler = CoreMLLogitsSampler
#endif
