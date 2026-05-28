import Foundation
import Testing
@testable import WatchLMCore

@Test func mockStreamingShortTurnBenchmark() async throws {
    let tokens = (0..<64).map { "t\($0)" }
    let runtime = MockStreamingRuntime(tokens: tokens)
    let iterations = 1_000
    let started = Date()

    for _ in 0..<iterations {
        let result = try await runtime.generate(
            request: InferenceRequest(prompt: "watch benchmark", maxNewTokens: 64),
            shouldCancel: { false }
        )
        #expect(result.tokens.count == 64)
    }

    let elapsedMs = Date().timeIntervalSince(started) * 1000
    let turnsPerSecond = Double(iterations) / max(Date().timeIntervalSince(started), 0.001)
    print("WATCHLM_SIM_BENCH mock_short_turn iterations=\(iterations) elapsed_ms=\(String(format: "%.3f", elapsedMs)) turns_per_second=\(String(format: "%.2f", turnsPerSecond))")
}
