import Foundation
import Testing
@testable import WatchLMCore

@Test func assetStatesRepresentInstallAndValidationLifecycle() throws {
    let manifest = try loadSampleManifest()
    let states: [ModelAssetState] = [
        .missing,
        .installing(progress: 0.5),
        .installed(manifest: manifest),
        .invalidHash(expected: "abc", actual: "def"),
        .incompatibleManifest(errors: ["runtime.type must be coreml-mlprogram"]),
        .unavailableRuntime(reason: "Core ML adapter unavailable")
    ]

    #expect(states.count == 6)
    #expect(states.contains(.missing))
    #expect(states.contains(.installing(progress: 0.5)))
    #expect(states.contains(.installed(manifest: manifest)))
}

@Test func assetStateIsCodableAndEquatable() throws {
    let state = ModelAssetState.invalidHash(expected: "abc", actual: "def")
    let encoded = try JSONEncoder().encode(state)
    let decoded = try JSONDecoder().decode(ModelAssetState.self, from: encoded)

    #expect(decoded == state)
}

@Test func modelAssetStoreReportsInstalledQwenStatefulApplicationSupportLayout() throws {
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
    let installedState = store.assetState(
        for: loadedManifest,
        deviceProfile: .watchSE2,
        runtimeCapabilities: CoreMLRuntimeCapabilities(
            platform: .watchOS,
            operatingSystemVersion: OperatingSystemVersion(majorVersion: 11, minorVersion: 0, patchVersion: 0)
        )
    )
    let selectedArtifact = try store.selectedArtifact(
        for: loadedManifest,
        deviceProfile: .watchSE2,
        requestedContextTokens: nil
    )

    #expect(loadedManifest.model.id == "Qwen/Qwen3-0.6B")
    #expect(installedState == .installed(manifest: loadedManifest))
    #expect(selectedArtifact.contextVariant == 256)
    #expect(selectedArtifact.prefillPath == "Models/Qwen3/stateful-step-kv-256-fp32-compute-int8.mlpackage")
    #expect(selectedArtifact.decodePath == selectedArtifact.prefillPath)
    #expect(store.url(for: selectedArtifact.prefillPath).standardizedFileURL.path() == statefulURL.standardizedFileURL.path())
}

@Test func modelAssetStoreReportsUnsupportedStatefulQwenOnWatchOS10() throws {
    let manifest = makeQwenStatefulTestManifest()
    let state = ModelAssetStore(rootURL: try makeTemporaryDirectory()).assetState(
        for: manifest,
        deviceProfile: .watchSE2,
        runtimeCapabilities: CoreMLRuntimeCapabilities(
            platform: .watchOS,
            operatingSystemVersion: OperatingSystemVersion(majorVersion: 10, minorVersion: 6, patchVersion: 0)
        )
    )

    if case .unavailableRuntime(let reason) = state {
        #expect(reason.contains("requires Core ML stateful prediction"))
    } else {
        Issue.record("Expected watchOS 10 to reject Qwen stateful-step runtime")
    }
}

@Test func modelAssetStoreBuildsQwenDeviceStagingPlanWithoutDuplicatingStatefulArtifact() throws {
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

    let plan = try store.stagingPlan(
        for: manifest,
        deviceProfile: .watchSE2
    )

    #expect(plan.deviceProfile == .watchSE2)
    #expect(plan.contextVariant == 256)
    #expect(plan.destinationRootDescription == "Application Support/WatchLM")
    #expect(plan.items.map(\.destinationRelativePath) == [
        "model-manifest.json",
        "Models/Qwen3/stateful-step-kv-256-fp32-compute-int8.mlpackage",
        "Models/Qwen3/tokenizer.json"
    ])
    #expect(plan.items[1].purposes == [.prefill, .decode])
    #expect(plan.items[1].expectedSHA256 == manifest.asset.prefillSHA256)
    #expect(plan.items[1].actualSHA256 == manifest.asset.prefillSHA256)
    #expect(plan.items[2].actualSHA256 == manifest.asset.tokenizerSHA256)
    #expect(plan.totalByteCount == plan.items.reduce(0) { $0 + $1.byteCount })
    #expect(plan.totalByteCount > 0)
}

@Test func modelAssetStagerCopiesQwenPlanAndTargetStoreVerifiesInstalledState() throws {
    let sourceRootURL = try makeTemporaryDirectory()
    let modelDirectory = sourceRootURL
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

    let sourceStore = ModelAssetStore(rootURL: sourceRootURL)
    try sourceStore.saveManifest(manifest)
    let plan = try sourceStore.stagingPlan(
        for: manifest,
        deviceProfile: .watchSE2
    )
    let targetRootURL = try makeTemporaryDirectory()

    let result = try ModelAssetStager().stage(
        plan: plan,
        to: targetRootURL
    )

    #expect(result.itemCount == 3)
    #expect(result.totalByteCount == plan.totalByteCount)
    #expect(FileManager.default.fileExists(atPath: targetRootURL.appending(path: "model-manifest.json").path))
    #expect(FileManager.default.fileExists(atPath: targetRootURL.appending(path: "Models/Qwen3/tokenizer.json").path))
    #expect(FileManager.default.fileExists(atPath: targetRootURL.appending(path: "Models/Qwen3/stateful-step-kv-256-fp32-compute-int8.mlpackage").path))

    let targetStore = ModelAssetStore(rootURL: targetRootURL)
    let targetManifest = try targetStore.loadManifest()
    let state = targetStore.assetState(
        for: targetManifest,
        deviceProfile: .watchSE2,
        runtimeCapabilities: CoreMLRuntimeCapabilities(
            platform: .watchOS,
            operatingSystemVersion: OperatingSystemVersion(majorVersion: 11, minorVersion: 0, patchVersion: 0)
        )
    )
    #expect(state == .installed(manifest: targetManifest))
}
