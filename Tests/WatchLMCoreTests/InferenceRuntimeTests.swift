import Testing
@testable import WatchLMCore

@Test func runtimeTimingRecordsLoadPrefillFirstTokenAndDecodeSteps() async throws {
    let runtime = MockStreamingRuntime(
        tokens: ["local", " answer"],
        loadMs: 7,
        prefillMs: 11,
        firstTokenMs: 19,
        decodeStepMs: [9, 10]
    )

    let loadTiming = try await runtime.load()
    let result = try await runtime.generate(
        request: InferenceRequest(prompt: "Hi", maxNewTokens: 2),
        shouldCancel: { false }
    )

    #expect(loadTiming.loadMs == 7)
    #expect(result.tokens == ["local", " answer"])
    #expect(result.text == "local answer")
    #expect(result.timing.prefillMs == 11)
    #expect(result.timing.firstTokenMs == 19)
    #expect(result.timing.decodeStepMs == [9, 10])
    #expect(result.timing.decodeTokensPerSecond == 105.26)
    #expect(result.timing.totalMs == 49)
}

@Test func mockRuntimeStopsAtMaxNewTokens() async throws {
    let runtime = MockStreamingRuntime(tokens: ["a", "b", "c"])

    let result = try await runtime.generate(
        request: InferenceRequest(prompt: "Hi", maxNewTokens: 2),
        shouldCancel: { false }
    )

    #expect(result.tokens == ["a", "b"])
}

@Test func cancellationIsObservedAtTokenBoundaries() async throws {
    let runtime = MockStreamingRuntime(tokens: ["one", "two", "three"])
    let probe = CancellationProbe(cancelOnCall: 2)

    do {
        _ = try await runtime.generate(
            request: InferenceRequest(prompt: "Hi", maxNewTokens: 3),
            shouldCancel: probe.shouldCancel
        )
        Issue.record("Expected cancellation")
    } catch let error as InferenceRuntimeError {
        #expect(error == .cancelled(partialTokens: ["one"]))
        #expect(error.userMessage == "Generation was cancelled.")
    }
}

@Test func runtimeErrorsAreTypedAndUserVisible() async throws {
    let missing = InferenceRuntimeError.modelAssetMissing
    let unavailable = InferenceRuntimeError.unavailableRuntime(reason: "Core ML adapter missing")

    #expect(missing.userMessage == "Model asset is not installed.")
    #expect(unavailable.userMessage == "Runtime unavailable: Core ML adapter missing")

    let runtime = MockStreamingRuntime(tokens: [], failure: missing)
    await #expect(throws: InferenceRuntimeError.modelAssetMissing) {
        _ = try await runtime.generate(
            request: InferenceRequest(prompt: "Hi", maxNewTokens: 1),
            shouldCancel: { false }
        )
    }
}

final class CancellationProbe: @unchecked Sendable {
    private let cancelOnCall: Int
    private var calls = 0

    init(cancelOnCall: Int) {
        self.cancelOnCall = cancelOnCall
    }

    func shouldCancel() -> Bool {
        calls += 1
        return calls >= cancelOnCall
    }
}
