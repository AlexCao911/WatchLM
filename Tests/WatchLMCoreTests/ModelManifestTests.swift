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

@Test func statefulStepCandidateManifestSelectsSharedSE2Artifact() throws {
    let manifest = try loadStatefulStepCandidateManifest()
    let artifact = try manifest.modelArtifact(for: .watchSE2, requestedContextTokens: nil)

    #expect(manifest.validationErrors.isEmpty)
    #expect(manifest.runtime.graphSchema.interface == "stateful-step-kv")
    #expect(artifact.contextVariant == 256)
    #expect(artifact.deviceProfile == "watch-se-2")
    #expect(artifact.prefillPath == "Models/MiniCPM5/stateful-step-kv-256-int4.mlpackage")
    #expect(artifact.decodePath == artifact.prefillPath)
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

@Test func manifestRuntimeGraphSchemaAcceptsStatefulKVInterface() throws {
    var manifest = try loadSampleManifest()
    manifest.runtime.graphSchema.interface = "stateful-kv"
    manifest.asset.prefillPath = "Models/MiniCPM5/stateful-512.mlpackage"
    manifest.asset.decodePath = "Models/MiniCPM5/stateful-512.mlpackage"
    manifest.asset.variants?["256"]?.prefillPath = "Models/MiniCPM5/stateful-256.mlpackage"
    manifest.asset.variants?["256"]?.decodePath = "Models/MiniCPM5/stateful-256.mlpackage"
    manifest.asset.variants?["512"]?.prefillPath = "Models/MiniCPM5/stateful-512.mlpackage"
    manifest.asset.variants?["512"]?.decodePath = "Models/MiniCPM5/stateful-512.mlpackage"

    let watchOS11 = CoreMLRuntimeCapabilities(
        platform: .watchOS,
        operatingSystemVersion: OperatingSystemVersion(majorVersion: 11, minorVersion: 0, patchVersion: 0)
    )

    #expect(manifest.validationErrors.isEmpty)
    #expect(manifest.runtime.kvCacheRouteDecision(capabilities: watchOS11).selectedRoute == .statefulKV)
}

@Test func manifestRuntimeGraphSchemaAcceptsStatefulStepKVInterface() throws {
    var manifest = try loadSampleManifest()
    manifest.runtime.graphSchema.interface = "stateful-step-kv"
    manifest.runtime.graphSchema.decode.tokenID = "input_ids"
    manifest.runtime.graphSchema.decode.positionID = "position_ids"
    manifest.asset.prefillPath = "Models/MiniCPM5/stateful-step-512.mlpackage"
    manifest.asset.decodePath = "Models/MiniCPM5/stateful-step-512.mlpackage"
    manifest.asset.variants?["256"]?.prefillPath = "Models/MiniCPM5/stateful-step-256.mlpackage"
    manifest.asset.variants?["256"]?.decodePath = "Models/MiniCPM5/stateful-step-256.mlpackage"
    manifest.asset.variants?["512"]?.prefillPath = "Models/MiniCPM5/stateful-step-512.mlpackage"
    manifest.asset.variants?["512"]?.decodePath = "Models/MiniCPM5/stateful-step-512.mlpackage"

    let watchOS11 = CoreMLRuntimeCapabilities(
        platform: .watchOS,
        operatingSystemVersion: OperatingSystemVersion(majorVersion: 11, minorVersion: 0, patchVersion: 0)
    )

    #expect(manifest.validationErrors.isEmpty)
    #expect(manifest.runtime.kvCacheRouteDecision(capabilities: watchOS11).selectedRoute == .statefulKV)
}

@Test func statefulGraphManifestRequiresSharedModelArtifacts() throws {
    var manifest = try loadSampleManifest()
    manifest.runtime.graphSchema.interface = "stateful-step-kv"
    manifest.runtime.graphSchema.decode.tokenID = "input_ids"
    manifest.runtime.graphSchema.decode.positionID = "position_ids"

    #expect(manifest.validationErrors.contains("stateful Core ML graphs must use the same artifact path for prefill and decode"))
    #expect(manifest.validationErrors.contains("asset.variants.256 must use the same artifact path for prefill and decode for stateful Core ML graphs"))
    #expect(manifest.validationErrors.contains("asset.variants.512 must use the same artifact path for prefill and decode for stateful Core ML graphs"))
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

@Test func coreMLBundleCanBeBuiltFromStatefulStepGraphSchema() throws {
    var manifest = try loadSampleManifest()
    manifest.runtime.graphSchema.interface = "stateful-step-kv"
    manifest.runtime.graphSchema.decode.tokenID = "input_ids"
    manifest.runtime.graphSchema.decode.positionID = "position_ids"

    let bundle = try CoreMLPrefillDecodeBundle(
        prefillModelURL: URL(fileURLWithPath: "/tmp/stateful-step.mlpackage"),
        decodeModelURL: URL(fileURLWithPath: "/tmp/stateful-step.mlpackage"),
        maxPromptTokens: 256,
        graphSchema: manifest.runtime.graphSchema
    )

    #expect(bundle.graphInterface == .statefulStepKV(layerCount: 24, kvHeads: 2, headDimension: 128))
    #expect(bundle.decodeTokenInputName == "input_ids")
    #expect(bundle.decodePositionInputName == "position_ids")
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
    #expect(assembly.kvCacheRouteDecision.selectedRoute == .explicitSlotRing)
    #expect(assembly.kvCacheRouteDecision.reason.contains("explicit KV tensors"))
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
    #expect(assembly.kvCacheRouteDecision.selectedRoute == .explicitContiguousSliding)
}

@Test func coreMLRuntimeAssemblerRejectsStatefulKVWhenRuntimeDoesNotSupportMLState() throws {
    let assetRoot = try makeTemporaryDirectory()
    let modelDirectory = assetRoot
        .appending(path: "Models", directoryHint: .isDirectory)
        .appending(path: "MiniCPM5", directoryHint: .isDirectory)
    try FileManager.default.createDirectory(at: modelDirectory, withIntermediateDirectories: true)

    let statefulURL = modelDirectory.appending(path: "stateful-256.mlpackage", directoryHint: .isDirectory)
    let tokenizerURL = modelDirectory.appending(path: "tokenizer.json")
    try FileManager.default.createDirectory(at: statefulURL, withIntermediateDirectories: true)
    try Data("stateful".utf8).write(to: statefulURL.appending(path: "Manifest.json"))
    try minimalTokenizerJSONData().write(to: tokenizerURL)

    var manifest = try loadSampleManifest()
    manifest.runtime.graphSchema.interface = "stateful-kv"
    manifest.asset.prefillPath = "Models/MiniCPM5/stateful-256.mlpackage"
    manifest.asset.decodePath = "Models/MiniCPM5/stateful-256.mlpackage"
    manifest.asset.variants?["256"]?.prefillPath = "Models/MiniCPM5/stateful-256.mlpackage"
    manifest.asset.variants?["256"]?.decodePath = "Models/MiniCPM5/stateful-256.mlpackage"
    manifest.asset.variants?["512"]?.prefillPath = "Models/MiniCPM5/stateful-512.mlpackage"
    manifest.asset.variants?["512"]?.decodePath = "Models/MiniCPM5/stateful-512.mlpackage"
    manifest.asset.variants?["256"]?.prefillSHA256 = try ArtifactDigest.sha256Hex(for: statefulURL)
    manifest.asset.variants?["256"]?.decodeSHA256 = try ArtifactDigest.sha256Hex(for: statefulURL)
    manifest.asset.variants?["256"]?.tokenizerSHA256 = try ArtifactDigest.sha256Hex(for: tokenizerURL)

    do {
        _ = try CoreMLRuntimeAssembler().assemble(
            manifest: manifest,
            deviceProfile: .watchSE2,
            assetBaseURL: assetRoot,
            runtimeCapabilities: CoreMLRuntimeCapabilities(
                platform: .watchOS,
                operatingSystemVersion: OperatingSystemVersion(majorVersion: 10, minorVersion: 6, patchVersion: 0)
            )
        )
        Issue.record("Expected assembler to reject unsupported stateful KV route")
    } catch CoreMLRuntimeAssemblyError.unsupportedRuntimeRoute(let decision) {
        #expect(decision.selectedRoute == .unsupportedStatefulKV)
    }
}
#endif

@Test func reportsManifestContractErrors() throws {
    var manifest = try loadSampleManifest()
    manifest.model.id = "wrong"
    manifest.runtime.type = "llama.cpp"
    manifest.runtime.kvCacheMode = "copy-everything"
    manifest.runtime.graphSchema.interface = "next-token"
    manifest.runtime.graphSchema.prefill.logits = "next_token"
    manifest.architecture.layers = 23

    #expect(manifest.validationErrors.contains("model.id must be openbmb/MiniCPM5-1B"))
    #expect(manifest.validationErrors.contains("runtime.type must be coreml-mlprogram"))
    #expect(manifest.validationErrors.contains("runtime.kvCacheMode must be stateful-preferred, slot-ring, or contiguous-sliding"))
    #expect(manifest.validationErrors.contains("runtime.graphSchema.interface must be logits-layered-kv, stateful-kv, or stateful-step-kv"))
    #expect(manifest.validationErrors.contains("runtime.graphSchema.prefill.logits must be logits"))
    #expect(manifest.validationErrors.contains("architecture.layers must be 24"))
}
