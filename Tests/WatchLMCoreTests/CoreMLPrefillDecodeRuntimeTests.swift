import Foundation
import Testing
@testable import WatchLMCore

#if canImport(CoreML)
@Test func coreMLPrefillDecodeRuntimeRunsSplitPredictionsWithExplicitKVCache() async throws {
    let bundle = CoreMLPrefillDecodeBundle(
        prefillModelURL: try #require(smokeModelURL(named: "SmokePrefill")),
        decodeModelURL: try #require(smokeModelURL(named: "SmokeDecode")),
        maxPromptTokens: 4
    )
    let runtime = CoreMLPrefillDecodeRuntime(
        bundle: bundle,
        tokenizer: FixtureTokenIDTokenizer()
    )

    let loadTiming = try await runtime.load()
    let result = try await runtime.generate(
        request: InferenceRequest(prompt: "A B", maxNewTokens: 3),
        shouldCancel: { false }
    )

    #expect(loadTiming.loadMs >= 0)
    #expect(result.tokens == ["D", "E", "F"])
    #expect(result.text == "DEF")
    #expect(result.timing.prefillMs >= 0)
    #expect(result.timing.decodeStepMs.count == 2)
    print("WATCHLM_PREFILL_DECODE_SMOKE output=\(result.text) load_ms=\(String(format: "%.3f", loadTiming.loadMs)) prefill_ms=\(String(format: "%.3f", result.timing.prefillMs)) decode_tps=\(String(format: "%.2f", result.timing.decodeTokensPerSecond))")
}

private func smokeModelURL(named baseName: String) -> URL? {
    #if os(watchOS)
    Bundle.module.url(forResource: "\(baseName)_watchOS", withExtension: "mlmodelc")
    #else
    Bundle.module.url(forResource: "\(baseName)_macOS", withExtension: "mlmodelc")
    #endif
}

private struct FixtureTokenIDTokenizer: TextTokenizer {
    let endOfSequenceTokenIDs: Set<Int32> = [1]

    func encode(_ text: String) throws -> [Int32] {
        try text
            .split(separator: " ")
            .map { piece in
                switch String(piece) {
                case "A": 2
                case "B": 3
                default:
                    throw InferenceRuntimeError.invalidInput(message: "Unknown fixture token \(piece).")
                }
            }
    }

    func decode(tokenIDs: [Int32]) throws -> String {
        try tokenIDs.map { tokenID in
            switch tokenID {
            case 5: "D"
            case 6: "E"
            case 7: "F"
            default:
                throw InferenceRuntimeError.predictionFailed(message: "Unknown fixture output token \(tokenID).")
            }
        }
        .joined()
    }
}
#endif
