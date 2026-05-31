#if canImport(XCTest)
import XCTest
@testable import WatchLMCore

final class WatchSimulatorAssetStoreXCTests: XCTestCase {
    func testQwenStatefulAssetStoreLayout() throws {
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
        let artifact = try store.selectedArtifact(for: loadedManifest, deviceProfile: .watchSE2)

        XCTAssertEqual(state, .installed(manifest: loadedManifest))
        XCTAssertEqual(artifact.contextVariant, 256)
        XCTAssertEqual(artifact.prefillPath, "Models/Qwen3/stateful-step-kv-256-fp32-compute-int8.mlpackage")
        XCTAssertEqual(artifact.decodePath, artifact.prefillPath)
        print("WATCHLM_XCTEST_QWEN_ASSET_STORE state=installed context=256 graph=stateful-step-kv")
    }
}
#endif
