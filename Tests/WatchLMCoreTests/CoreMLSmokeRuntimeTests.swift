import Foundation
import Testing
@testable import WatchLMCore

#if canImport(CoreML)
@Test func coreMLSmokeRuntimeRunsRealPrediction() async throws {
    let modelURL = try #require(smokeModelURL())
    let runtime = CoreMLSmokeRuntime(
        modelURL: modelURL,
        inputName: "token",
        outputName: "logits"
    )

    let loadTiming = try await runtime.load()
    let result = try await runtime.generate(
        request: InferenceRequest(prompt: "7.5", maxNewTokens: 1),
        shouldCancel: { false }
    )

    #expect(loadTiming.loadMs >= 0)
    #expect(result.tokens == ["7.5"])
    #expect(result.text == "7.5")
    #expect(result.timing.firstTokenMs >= 0)
    #expect(result.timing.decodeStepMs.count == 1)
    print("WATCHLM_COREML_SMOKE output=\(result.text) load_ms=\(String(format: "%.3f", loadTiming.loadMs)) prediction_ms=\(String(format: "%.3f", result.timing.firstTokenMs))")
}

private func smokeModelURL() -> URL? {
    #if os(watchOS)
    Bundle.module.url(forResource: "SmokeIdentity_watchOS", withExtension: "mlmodelc")
    #else
    Bundle.module.url(forResource: "SmokeIdentity_macOS", withExtension: "mlmodelc")
    #endif
}
#endif
