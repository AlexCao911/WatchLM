import Foundation
import Testing
@testable import WatchLMCore

@Test func decodesSampleModelManifest() throws {
    let manifest = try loadSampleManifest()

    #expect(manifest.model.id == "openbmb/MiniCPM5-1B")
    #expect(manifest.runtime.type == "coreml-mlprogram")
    #expect(manifest.architecture.layers == 24)
    #expect(manifest.architecture.hiddenSize == 1536)
    #expect(manifest.architecture.queryHeads == 16)
    #expect(manifest.architecture.kvHeads == 2)
    #expect(manifest.architecture.tokenizer.preserved)
    #expect(manifest.architecture.tokenizer.vocabularyPreserved)
    #expect(manifest.contextVariants == [256, 512, 1024])
    #expect(manifest.validationErrors.isEmpty)
}

@Test func reportsManifestContractErrors() throws {
    var manifest = try loadSampleManifest()
    manifest.model.id = "wrong"
    manifest.runtime.type = "llama.cpp"
    manifest.architecture.layers = 23

    #expect(manifest.validationErrors.contains("model.id must be openbmb/MiniCPM5-1B"))
    #expect(manifest.validationErrors.contains("runtime.type must be coreml-mlprogram"))
    #expect(manifest.validationErrors.contains("architecture.layers must be 24"))
}

