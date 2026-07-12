import Testing
@testable import WatchLMCore

@Test func greedySamplerSelectsHighestLogitAndUsesLowestTokenIDForTies() throws {
    let sampler = GreedyTokenSampler()

    let tokenID = try sampler.selectToken(from: [
        TokenLogit(tokenID: 42, logit: 1.5),
        TokenLogit(tokenID: 7, logit: 3.0),
        TokenLogit(tokenID: 5, logit: 3.0)
    ])

    #expect(tokenID == 5)
}

@Test func seededSamplerProducesRepeatableProbabilityWeightedSequence() throws {
    let logits = [
        TokenLogit(tokenID: 10, logit: 0),
        TokenLogit(tokenID: 11, logit: 0),
        TokenLogit(tokenID: 12, logit: 0),
        TokenLogit(tokenID: 13, logit: 0)
    ]
    let left = SeededTokenSampler(seed: 0xC0FFEE)
    let right = SeededTokenSampler(seed: 0xC0FFEE)

    let leftTokens = try (0..<12).map { _ in try left.selectToken(from: logits) }
    let rightTokens = try (0..<12).map { _ in try right.selectToken(from: logits) }

    #expect(leftTokens == rightTokens)
    #expect(Set(leftTokens).count > 1)
    #expect(leftTokens.contains { $0 != 10 })
}

@Test func samplingStrategyBuildsTheRequestedSampler() throws {
    let logits = [
        TokenLogit(tokenID: 1, logit: 0),
        TokenLogit(tokenID: 2, logit: 0),
        TokenLogit(tokenID: 3, logit: 0)
    ]

    let first = TokenSamplingStrategy.seeded(seed: 42).makeSampler()
    let second = TokenSamplingStrategy.seeded(seed: 42).makeSampler()

    let firstTokens = try (0..<8).map { _ in try first.selectToken(from: logits) }
    let secondTokens = try (0..<8).map { _ in try second.selectToken(from: logits) }

    #expect(try TokenSamplingStrategy.greedy.makeSampler().selectToken(from: logits) == 1)
    #expect(firstTokens == secondTokens)
    #expect(Set(firstTokens).count > 1)
}

@Test func logitsProcessorAppliesRepetitionPenaltyTemperatureTopKAndTopP() throws {
    let processor = LogitsProcessor(
        temperature: 2.0,
        topK: 3,
        topP: 0.78,
        repetitionPenalty: 2.0
    )

    let processed = try processor.process(
        logits: [
            TokenLogit(tokenID: 1, logit: 8.0),
            TokenLogit(tokenID: 2, logit: 6.0),
            TokenLogit(tokenID: 3, logit: 4.0),
            TokenLogit(tokenID: 4, logit: 2.0)
        ],
        generatedTokenIDs: [1]
    )

    #expect(processed == [
        TokenLogit(tokenID: 2, logit: 3.0),
        TokenLogit(tokenID: 1, logit: 2.0)
    ])
}

@Test func logitsProcessorRejectsInvalidSamplingConfiguration() {
    #expect(throws: InferenceRuntimeError.invalidInput(message: "temperature must be greater than zero.")) {
        _ = try LogitsProcessor(temperature: 0).process(
            logits: [TokenLogit(tokenID: 1, logit: 1.0)],
            generatedTokenIDs: []
        )
    }

    #expect(throws: InferenceRuntimeError.invalidInput(message: "topP must be in the range (0, 1].")) {
        _ = try LogitsProcessor(topP: 1.5).process(
            logits: [TokenLogit(tokenID: 1, logit: 1.0)],
            generatedTokenIDs: []
        )
    }
}

@Test func decodeStopCriteriaStopsAtEOSOrMaxNewTokens() {
    let criteria = DecodeStopCriteria(maxNewTokens: 4, eosTokenIDs: [1, 130073])

    #expect(!criteria.shouldStop(generatedTokenIDs: [12, 13]))
    #expect(criteria.shouldStop(generatedTokenIDs: [12, 130073]))
    #expect(criteria.shouldStop(generatedTokenIDs: [12, 13, 14, 15]))
}
