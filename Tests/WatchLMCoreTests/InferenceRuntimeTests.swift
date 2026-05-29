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
    #expect(result.generatedTokenIDs == [])
    #expect(result.text == "local answer")
    #expect(result.terminationReason == .maxTokens)
    #expect(result.timing.prefillMs == 11)
    #expect(result.timing.firstTokenMs == 19)
    #expect(result.timing.decodeStepMs == [9, 10])
    #expect(result.timing.decodeTokensPerSecond == 105.26)
    #expect(result.timing.totalMs == 49)
    #expect(result.metrics == InferenceMetrics())
}

@Test func mockRuntimeStreamsTokenEventsBeforeCompletion() async throws {
    let runtime = MockStreamingRuntime(
        tokens: ["local", " answer"],
        prefillMs: 4,
        firstTokenMs: 7,
        decodeStepMs: [7, 3]
    )

    var events: [InferenceStreamEvent] = []
    for try await event in runtime.stream(
        request: InferenceRequest(prompt: "Hi", maxNewTokens: 2),
        shouldCancel: { false }
    ) {
        events.append(event)
    }

    #expect(events == [
        .token(InferenceToken(index: 0, tokenID: nil, text: "local", isFirstToken: true)),
        .token(InferenceToken(index: 1, tokenID: nil, text: " answer", isFirstToken: false)),
        .completed(
            InferenceResult(
                tokens: ["local", " answer"],
                timing: RuntimeTiming(
                    prefillMs: 4,
                    firstTokenMs: 7,
                    decodeStepMs: [7, 3]
                ),
                terminationReason: .maxTokens
            )
        )
    ])
}

@Test func mockRuntimeStreamCancellationPreservesPartialTokens() async throws {
    let runtime = MockStreamingRuntime(tokens: ["one", "two", "three"])
    let probe = CancellationProbe(cancelOnCall: 2)

    var events: [InferenceStreamEvent] = []
    do {
        for try await event in runtime.stream(
            request: InferenceRequest(prompt: "Hi", maxNewTokens: 3),
            shouldCancel: probe.shouldCancel
        ) {
            events.append(event)
        }
        Issue.record("Expected streaming cancellation")
    } catch let error as InferenceRuntimeError {
        #expect(events == [
            .token(InferenceToken(index: 0, tokenID: nil, text: "one", isFirstToken: true))
        ])
        #expect(error == .cancelled(partialTokens: ["one"]))
    }
}

@Test func inferenceMetricsSummarizeDecodeComponentWork() {
    let metrics = InferenceMetrics(
        prefillSamplingMs: 0.4,
        decodeSamplingStepMs: [0.2, 0.3],
        kvAppendStepMs: [0.5, 0.6],
        kvCacheUpdateStrategy: .slotRing,
        kvAppendWriteIndices: [1, 0],
        kvAppendMovedTokenSlots: [2, 3],
        kvAppendMovedScalarCounts: [4, 6]
    )

    #expect(metrics.totalDecodeSamplingMs == 0.5)
    #expect(metrics.totalKVAppendMs == 1.1)
    #expect(metrics.kvCacheUpdateStrategy == .slotRing)
    #expect(metrics.kvAppendWriteIndices == [1, 0])
    #expect(metrics.totalKVAppendMovedTokenSlots == 5)
    #expect(metrics.totalKVAppendMovedScalarCount == 10)
}

@Test func mockRuntimeStopsAtMaxNewTokens() async throws {
    let runtime = MockStreamingRuntime(tokens: ["a", "b", "c"])

    let result = try await runtime.generate(
        request: InferenceRequest(prompt: "Hi", maxNewTokens: 2),
        shouldCancel: { false }
    )

    #expect(result.tokens == ["a", "b"])
    #expect(result.terminationReason == .maxTokens)
}

@Test func mockRuntimeReportsSourceExhaustedWhenFixtureEndsBeforeMaxTokens() async throws {
    let runtime = MockStreamingRuntime(tokens: ["a"])

    let result = try await runtime.generate(
        request: InferenceRequest(prompt: "Hi", maxNewTokens: 3),
        shouldCancel: { false }
    )

    #expect(result.tokens == ["a"])
    #expect(result.terminationReason == .sourceExhausted)
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
