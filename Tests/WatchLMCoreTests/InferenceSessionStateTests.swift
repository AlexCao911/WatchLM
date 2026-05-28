import Foundation
import Testing
@testable import WatchLMCore

@Test func inferenceSessionStatesRepresentShortForegroundLifecycle() {
    let states: [InferenceSessionState] = [
        .idle,
        .prefill,
        .decoding(generatedTokens: 1),
        .cancelled,
        .finished(totalTokens: 32),
        .failed(message: "model asset missing"),
        .thermalDegraded
    ]

    #expect(states.count == 7)
    #expect(states.contains(.idle))
    #expect(states.contains(.decoding(generatedTokens: 1)))
    #expect(states.contains(.thermalDegraded))
}

@Test func terminalSessionStatesAreExplicit() {
    #expect(!InferenceSessionState.idle.isTerminal)
    #expect(!InferenceSessionState.prefill.isTerminal)
    #expect(!InferenceSessionState.decoding(generatedTokens: 12).isTerminal)
    #expect(InferenceSessionState.cancelled.isTerminal)
    #expect(InferenceSessionState.finished(totalTokens: 12).isTerminal)
    #expect(InferenceSessionState.failed(message: "no model").isTerminal)
    #expect(InferenceSessionState.thermalDegraded.isTerminal)
}

