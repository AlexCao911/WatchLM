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

@Test func decodeStopCriteriaStopsAtEOSOrMaxNewTokens() {
    let criteria = DecodeStopCriteria(maxNewTokens: 4, eosTokenIDs: [1, 130073])

    #expect(!criteria.shouldStop(generatedTokenIDs: [12, 13]))
    #expect(criteria.shouldStop(generatedTokenIDs: [12, 130073]))
    #expect(criteria.shouldStop(generatedTokenIDs: [12, 13, 14, 15]))
}
