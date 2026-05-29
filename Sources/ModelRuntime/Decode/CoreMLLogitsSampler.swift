#if canImport(CoreML)
import CoreML

struct CoreMLLogitsProcessor {
    var policy: LogitsProcessor

    init(topK: Int? = nil) {
        policy = LogitsProcessor(topK: topK)
    }

    init(policy: LogitsProcessor) {
        self.policy = policy
    }

    func tokenLogits(
        from logits: MLMultiArray,
        generatedTokenIDs: [Int32] = []
    ) throws -> [TokenLogit] {
        guard logits.count > 0 else {
            throw InferenceRuntimeError.predictionFailed(message: "Logits output is empty.")
        }

        let tokenLogits = (0..<logits.count).map { tokenID in
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
