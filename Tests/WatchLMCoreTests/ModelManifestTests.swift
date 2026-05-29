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
    #expect(manifest.asset.variants?["256"]?.deviceProfile == "watch-se-2")
    #expect(manifest.asset.variants?["512"]?.deviceProfile == "watch-se-3")
    #expect(manifest.validationErrors.isEmpty)
}

@Test func selectsModelArtifactForSE2AndSE3() throws {
    let manifest = try loadSampleManifest()

    let se2Artifact = try manifest.modelArtifact(
        for: .watchSE2,
        requestedContextTokens: nil
    )
    let se3Artifact = try manifest.modelArtifact(
        for: .watchSE3,
        requestedContextTokens: nil
    )

    #expect(se2Artifact.contextVariant == 256)
    #expect(se2Artifact.prefillPath == "Models/MiniCPM5/prefill-256.mlpackage")
    #expect(se2Artifact.decodePath == "Models/MiniCPM5/decode-256.mlpackage")
    #expect(se2Artifact.tokenizerPath == "Models/MiniCPM5/tokenizer.json")
    #expect(se2Artifact.prefillSHA256 == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    #expect(se2Artifact.decodeSHA256 == "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    #expect(se2Artifact.tokenizerSHA256 == "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc")
    #expect(se3Artifact.contextVariant == 512)
    #expect(se3Artifact.prefillPath == "Models/MiniCPM5/prefill-512.mlpackage")
    #expect(se3Artifact.decodePath == "Models/MiniCPM5/decode-512.mlpackage")
    #expect(se3Artifact.tokenizerPath == "Models/MiniCPM5/tokenizer.json")
    #expect(se3Artifact.prefillSHA256 == "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd")
    #expect(se3Artifact.decodeSHA256 == "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
    #expect(se3Artifact.tokenizerSHA256 == "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc")
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
