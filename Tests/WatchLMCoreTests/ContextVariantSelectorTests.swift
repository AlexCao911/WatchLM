import Testing
@testable import WatchLMCore

@Test func selectsDefaultDeviceContext() throws {
    let manifest = try loadSampleManifest()

    #expect(try ContextVariantSelector.select(from: manifest, for: .watchSE2) == 256)
    #expect(try ContextVariantSelector.select(from: manifest, for: .watchSE3) == 512)
}

@Test func clampsRequestedContextToSupportedVariant() throws {
    let manifest = try loadSampleManifest()

    #expect(try ContextVariantSelector.select(from: manifest, for: .watchSE3, requestedTokens: 1024) == 1024)
    #expect(try ContextVariantSelector.select(from: manifest, for: .watchSE3, requestedTokens: 999) == 512)
    #expect(try ContextVariantSelector.select(from: manifest, for: .watchSE3, requestedTokens: 128) == 256)
}

@Test func rejectsMissingDeviceProfile() throws {
    var manifest = try loadSampleManifest()
    manifest.deviceProfiles.removeValue(forKey: "watch-se-2")

    #expect(throws: ContextVariantSelectionError.unsupportedDeviceProfile("watch-se-2")) {
        try ContextVariantSelector.select(from: manifest, for: .watchSE2)
    }
}

