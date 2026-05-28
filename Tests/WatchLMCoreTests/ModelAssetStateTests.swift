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

