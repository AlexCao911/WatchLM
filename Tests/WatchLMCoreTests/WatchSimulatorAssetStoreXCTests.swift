#if canImport(XCTest)
import Foundation
import XCTest
@testable import WatchLMCore
#if canImport(CoreML)
import CoreML
#endif

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

    func testQwenRealCoreMLLoadOnly() throws {
        let environment = ProcessInfo.processInfo.environment
        let sentinelURL = URL(fileURLWithPath: "/private/tmp/watchlm-run-qwen-se2-load")
        let shouldRun = environment["WATCHLM_RUN_REAL_QWEN_SE2_LOAD"] == "1"
            || FileManager.default.fileExists(atPath: sentinelURL.path)
        guard shouldRun else {
            throw XCTSkip("Set WATCHLM_RUN_REAL_QWEN_SE2_LOAD=1 or create \(sentinelURL.path) to run the real Qwen Core ML load gate.")
        }

        #if canImport(CoreML)
        let modelURL = qwenRealCoreMLModelURL(environment: environment)
        XCTAssertTrue(FileManager.default.fileExists(atPath: modelURL.path), "Missing model at \(modelURL.path)")

        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        let started = Date()
        _ = try MLModel(contentsOf: modelURL, configuration: configuration)
        let loadMs = Date().timeIntervalSince(started) * 1000
        print(
            "WATCHLM_XCTEST_QWEN_REAL_LOAD result=loaded load_ms=\(String(format: "%.3f", loadMs)) model=\(modelURL.lastPathComponent)"
        )
        #else
        throw XCTSkip("Core ML is unavailable in this test environment.")
        #endif
    }

    private func qwenRealCoreMLModelURL(environment: [String: String]) -> URL {
        if let modelPath = environment["WATCHLM_QWEN_MODEL_URL"], !modelPath.isEmpty {
            return URL(fileURLWithPath: modelPath)
        }

        return repositoryRootURL()
            .appending(path: "artifacts", directoryHint: .isDirectory)
            .appending(path: "coreml", directoryHint: .isDirectory)
            .appending(path: "compiled-watchos11-qwen3-0.6b-stateful-step-kv-256-fp32-compute-int8", directoryHint: .isDirectory)
            .appending(path: "stateful-step-kv-256-int8.mlmodelc", directoryHint: .isDirectory)
    }

    private func repositoryRootURL() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }
}
#endif
