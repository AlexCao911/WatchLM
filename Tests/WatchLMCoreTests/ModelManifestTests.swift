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

@Test func manifestRuntimeGraphSchemaMatchesExplicitKVCoreMLIO() throws {
    let manifest = try loadSampleManifest()
    let schema = manifest.runtime.graphSchema

    #expect(schema.interface == "logits-layered-kv")
    #expect(schema.layerCount == 24)
    #expect(schema.kvHeads == 2)
    #expect(schema.headDimension == 128)
    #expect(schema.prefill.inputIDs == "input_ids")
    #expect(schema.prefill.positionIDs == "position_ids")
    #expect(schema.prefill.causalMask == "causal_mask")
    #expect(schema.prefill.logits == "logits")
    #expect(schema.prefill.keyPrefix == "present_key_")
    #expect(schema.prefill.valuePrefix == "present_value_")
    #expect(schema.decode.tokenID == "token_id")
    #expect(schema.decode.positionID == "position_id")
    #expect(schema.decode.causalMask == "causal_mask")
    #expect(schema.decode.logits == "logits")
    #expect(schema.decode.pastKeyPrefix == "past_key_")
    #expect(schema.decode.pastValuePrefix == "past_value_")
    #expect(schema.decode.newKeyPrefix == "new_key_")
    #expect(schema.decode.newValuePrefix == "new_value_")
}

#if canImport(CoreML)
@Test func coreMLBundleCanBeBuiltFromManifestGraphSchema() throws {
    let manifest = try loadSampleManifest()

    let bundle = try CoreMLPrefillDecodeBundle(
        prefillModelURL: URL(fileURLWithPath: "/tmp/prefill.mlpackage"),
        decodeModelURL: URL(fileURLWithPath: "/tmp/decode.mlpackage"),
        maxPromptTokens: 256,
        graphSchema: manifest.runtime.graphSchema
    )

    #expect(bundle.graphInterface == .logitsAndLayeredKV(layerCount: 24, kvHeads: 2, headDimension: 128))
    #expect(bundle.prefillInputName == "input_ids")
    #expect(bundle.prefillPositionInputName == "position_ids")
    #expect(bundle.prefillCausalMaskInputName == "causal_mask")
    #expect(bundle.prefillLogitsOutputName == "logits")
    #expect(bundle.decodeTokenInputName == "token_id")
    #expect(bundle.decodePositionInputName == "position_id")
    #expect(bundle.decodeCausalMaskInputName == "causal_mask")
    #expect(bundle.decodeLogitsOutputName == "logits")
    #expect(bundle.decodePastKeyInputName(forLayer: 23) == "past_key_23")
    #expect(bundle.decodeNewValueOutputName(forLayer: 23) == "new_value_23")
}

@Test func coreMLRuntimeAssemblerBuildsManifestSelectedRuntimeComponents() throws {
    let assetRoot = try makeTemporaryDirectory()
    let modelDirectory = assetRoot
        .appending(path: "Models", directoryHint: .isDirectory)
        .appending(path: "MiniCPM5", directoryHint: .isDirectory)
    try FileManager.default.createDirectory(at: modelDirectory, withIntermediateDirectories: true)

    let prefillURL = modelDirectory.appending(path: "prefill-256.mlpackage", directoryHint: .isDirectory)
    let decodeURL = modelDirectory.appending(path: "decode-256.mlpackage", directoryHint: .isDirectory)
    let tokenizerURL = modelDirectory.appending(path: "tokenizer.json")
    try FileManager.default.createDirectory(at: prefillURL, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: decodeURL, withIntermediateDirectories: true)
    try Data("prefill".utf8).write(to: prefillURL.appending(path: "Manifest.json"))
    try Data("decode".utf8).write(to: decodeURL.appending(path: "Manifest.json"))
    try minimalTokenizerJSONData().write(to: tokenizerURL)

    var manifest = try loadSampleManifest()
    manifest.asset.variants?["256"]?.prefillSHA256 = try ArtifactDigest.sha256Hex(for: prefillURL)
    manifest.asset.variants?["256"]?.decodeSHA256 = try ArtifactDigest.sha256Hex(for: decodeURL)
    manifest.asset.variants?["256"]?.tokenizerSHA256 = try ArtifactDigest.sha256Hex(for: tokenizerURL)

    let assembly = try CoreMLRuntimeAssembler().assemble(
        manifest: manifest,
        deviceProfile: .watchSE2,
        requestedContextTokens: nil,
        assetBaseURL: assetRoot,
        samplingStrategy: .seeded(seed: 999)
    )

    #expect(assembly.artifact.contextVariant == 256)
    #expect(assembly.verificationReport.isReady)
    #expect(assembly.artifact.deviceProfile == "watch-se-2")
    #expect(assembly.prefillModelURL == prefillURL)
    #expect(assembly.decodeModelURL == decodeURL)
    #expect(assembly.tokenizerURL == tokenizerURL)
    #expect(assembly.bundle.maxPromptTokens == 256)
    #expect(assembly.bundle.prefillInputName == "input_ids")
    #expect(assembly.bundle.decodeTokenInputName == "token_id")
    #expect(assembly.bundle.kvCacheUpdateStrategy == .slotRing)
    #expect(assembly.bundle.samplingStrategy == .seeded(seed: 999))
    #expect(try assembly.tokenizer.encode("Hi") == [0, 19301])
    _ = assembly.makeRuntime()
}

@Test func coreMLRuntimeAssemblerMapsManifestKVCacheModeToUpdateStrategy() throws {
    let assetRoot = try makeTemporaryDirectory()
    let modelDirectory = assetRoot
        .appending(path: "Models", directoryHint: .isDirectory)
        .appending(path: "MiniCPM5", directoryHint: .isDirectory)
    try FileManager.default.createDirectory(at: modelDirectory, withIntermediateDirectories: true)

    let prefillURL = modelDirectory.appending(path: "prefill-256.mlpackage", directoryHint: .isDirectory)
    let decodeURL = modelDirectory.appending(path: "decode-256.mlpackage", directoryHint: .isDirectory)
    let tokenizerURL = modelDirectory.appending(path: "tokenizer.json")
    try FileManager.default.createDirectory(at: prefillURL, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: decodeURL, withIntermediateDirectories: true)
    try Data("prefill".utf8).write(to: prefillURL.appending(path: "Manifest.json"))
    try Data("decode".utf8).write(to: decodeURL.appending(path: "Manifest.json"))
    try minimalTokenizerJSONData().write(to: tokenizerURL)

    var manifest = try loadSampleManifest()
    manifest.runtime.kvCacheMode = "contiguous-sliding"
    manifest.asset.variants?["256"]?.prefillSHA256 = try ArtifactDigest.sha256Hex(for: prefillURL)
    manifest.asset.variants?["256"]?.decodeSHA256 = try ArtifactDigest.sha256Hex(for: decodeURL)
    manifest.asset.variants?["256"]?.tokenizerSHA256 = try ArtifactDigest.sha256Hex(for: tokenizerURL)

    let assembly = try CoreMLRuntimeAssembler().assemble(
        manifest: manifest,
        deviceProfile: .watchSE2,
        requestedContextTokens: nil,
        assetBaseURL: assetRoot
    )

    #expect(assembly.bundle.kvCacheUpdateStrategy == .contiguousSliding)
}
#endif

@Test func reportsManifestContractErrors() throws {
    var manifest = try loadSampleManifest()
    manifest.model.id = "wrong"
    manifest.runtime.type = "llama.cpp"
    manifest.runtime.kvCacheMode = "copy-everything"
    manifest.runtime.graphSchema.prefill.logits = "next_token"
    manifest.architecture.layers = 23

    #expect(manifest.validationErrors.contains("model.id must be openbmb/MiniCPM5-1B"))
    #expect(manifest.validationErrors.contains("runtime.type must be coreml-mlprogram"))
    #expect(manifest.validationErrors.contains("runtime.kvCacheMode must be stateful-preferred, slot-ring, or contiguous-sliding"))
    #expect(manifest.validationErrors.contains("runtime.graphSchema.prefill.logits must be logits"))
    #expect(manifest.validationErrors.contains("architecture.layers must be 24"))
}
