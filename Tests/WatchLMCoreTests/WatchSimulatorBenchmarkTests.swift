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

@Test func qwenStatefulAssetStoreLayoutBenchmark() throws {
    let rootURL = try makeTemporaryDirectory()
    let modelDirectory = rootURL
        .appending(path: "Models", directoryHint: .isDirectory)
        .appending(path: "Qwen3", directoryHint: .isDirectory)
    let statefulURL = modelDirectory
        .appending(path: "stateful-step-kv-256-fp32-compute-int8.mlpackage", directoryHint: .isDirectory)
    let tokenizerURL = modelDirectory.appending(path: "tokenizer.json")
    try FileManager.default.createDirectory(at: statefulURL, withIntermediateDirectories: true)
    try Data("qwen-stateful".utf8).write(to: statefulURL.appending(path: "Manifest.json"))
    try minimalTokenizerJSONData().write(to: tokenizerURL)

    var manifest = makeQwenStatefulTestManifest()
    manifest.asset.prefillSHA256 = try ArtifactDigest.sha256Hex(for: statefulURL)
    manifest.asset.decodeSHA256 = try ArtifactDigest.sha256Hex(for: statefulURL)
    manifest.asset.tokenizerSHA256 = try ArtifactDigest.sha256Hex(for: tokenizerURL)

    let store = ModelAssetStore(rootURL: rootURL)
    try store.saveManifest(manifest)
    let loadedManifest = try store.loadManifest()
    let state = store.assetState(
        for: loadedManifest,
        deviceProfile: .watchSE2,
        runtimeCapabilities: CoreMLRuntimeCapabilities(
            platform: .watchOS,
            operatingSystemVersion: OperatingSystemVersion(majorVersion: 11, minorVersion: 0, patchVersion: 0)
        )
    )

    #expect(state == .installed(manifest: loadedManifest))
    #expect(try store.selectedArtifact(for: loadedManifest, deviceProfile: .watchSE2).contextVariant == 256)
    print("WATCHLM_SIM_QWEN_ASSET_STORE state=installed context=256 graph=stateful-step-kv")
}
