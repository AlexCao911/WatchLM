import Foundation
import Testing
@testable import WatchLMCore

@Test func mixedPrecisionPolicyProtectsSensitiveTransformerLayers() throws {
    let manifest = try loadSampleManifest()
    let policy = try MixedPrecisionPolicy(manifest: manifest)

    #expect(policy.embedding == .int8)
    #expect(policy.lmHead == .int8)
    #expect(policy.norms == .fp16)
    #expect(policy.kvCache == .int8)
    #expect(policy.shouldProtectTransformerLayer(0))
    #expect(policy.shouldProtectTransformerLayer(1))
    #expect(!policy.shouldProtectTransformerLayer(2))
    #expect(!policy.shouldProtectTransformerLayer(21))
    #expect(policy.shouldProtectTransformerLayer(22))
    #expect(policy.shouldProtectTransformerLayer(23))
    #expect(policy.precision(for: .ffn, layer: 12) == .int4)
    #expect(policy.precision(for: .ffn, layer: 0) == .int8)
    #expect(policy.precision(for: .attentionV, layer: 23) == .int8)
    #expect(policy.precision(for: .attentionQKO, layer: 12) == .int8)
}

@Test func mixedPrecisionPolicyAllowsFloat16KVCacheForFidelityProfiles() throws {
    var manifest = try loadSampleManifest()
    manifest.quantization.kvCache = "fp16"

    let policy = try MixedPrecisionPolicy(manifest: manifest)

    #expect(policy.kvCache == .fp16)
    #expect(policy.kvCacheDescriptorPrecision == .float16)
}

@Test func mixedPrecisionPolicyAppliesLayerOverridesBeforeEdgeProtection() throws {
    var manifest = try loadSampleManifest()
    manifest.quantization.weights.attentionV = "fp16"
    manifest.quantization.layerOverrides = [
        .attentionV: [5: "int4", 6: "int4"],
        .ffn: [0: "int4", 12: "int8"]
    ]

    let policy = try MixedPrecisionPolicy(manifest: manifest)

    #expect(policy.precision(for: .attentionV, layer: 5) == .int4)
    #expect(policy.precision(for: .attentionV, layer: 6) == .int4)
    #expect(policy.precision(for: .attentionV, layer: 7) == .fp16)
    #expect(policy.precision(for: .ffn, layer: 12) == .int8)
    #expect(policy.precision(for: .ffn, layer: 0) == .int8)
    #expect(policy.precision(for: .ffn, layer: 23) == .int8)
}

@Test func mixedPrecisionPolicyCanControlFFNSubcomponents() throws {
    var manifest = try loadSampleManifest()
    manifest.quantization.weights.ffn = "fp16"
    manifest.quantization.weights.ffnGateUp = "int8"
    manifest.quantization.weights.ffnDown = "fp16"
    manifest.quantization.layerOverrides = [
        .ffnGateUp: [12: "int4"],
        .ffnDown: [12: "int8"]
    ]

    let policy = try MixedPrecisionPolicy(
        manifest: manifest,
        protectedEdgeLayerCount: 0
    )

    #expect(policy.precision(for: .ffn, layer: 12) == .fp16)
    #expect(policy.precision(for: .ffnGateUp, layer: 12) == .int4)
    #expect(policy.precision(for: .ffnDown, layer: 12) == .int8)
    #expect(policy.precision(for: .ffnGateUp, layer: 11) == .int8)
    #expect(policy.precision(for: .ffnDown, layer: 11) == .fp16)
}

@Test func mixedPrecisionPolicyConsumesRealImportanceGuidedPolicyFile() throws {
    let url = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        .appending(path: "tools/conversion/mixed-precision-policy-stateful-step-importance-attention-v-low4-int4.json")
    let quantization = try JSONDecoder().decode(QuantizationInfo.self, from: Data(contentsOf: url))

    let policy = try MixedPrecisionPolicy(
        quantization: quantization,
        layerCount: 24,
        protectedEdgeLayerCount: 0
    )

    #expect(policy.precision(for: .attentionV, layer: 5) == .int4)
    #expect(policy.precision(for: .attentionV, layer: 6) == .int4)
    #expect(policy.precision(for: .attentionV, layer: 7) == .int4)
    #expect(policy.precision(for: .attentionV, layer: 8) == .int4)
    #expect(policy.precision(for: .attentionV, layer: 9) == .fp16)
    #expect(policy.precision(for: .attentionQKO, layer: 6) == .fp16)
    #expect(policy.precision(for: .ffn, layer: 6) == .fp16)
    #expect(policy.kvCache == .fp16)
}

@Test func mixedPrecisionPolicyRejectsUniformLowBitOrStructuralReduction() throws {
    var manifest = try loadSampleManifest()
    manifest.quantization.strategy = "uniform-int4"

    #expect(throws: MixedPrecisionPolicyError.unsupportedStrategy("uniform-int4")) {
        _ = try MixedPrecisionPolicy(manifest: manifest)
    }

    manifest = try loadSampleManifest()
    manifest.quantization.structuralReduction = true

    #expect(throws: MixedPrecisionPolicyError.structuralReductionEnabled) {
        _ = try MixedPrecisionPolicy(manifest: manifest)
    }
}

@Test func mixedPrecisionPolicyRejectsUnsupportedPrecisionStrings() throws {
    var manifest = try loadSampleManifest()
    manifest.quantization.weights.ffn = "int2"

    #expect(throws: MixedPrecisionPolicyError.unsupportedPrecision("int2")) {
        _ = try MixedPrecisionPolicy(manifest: manifest)
    }

    manifest = try loadSampleManifest()
    manifest.quantization.kvCache = "int4"

    #expect(throws: MixedPrecisionPolicyError.unsupportedKVCachePrecision("int4")) {
        _ = try MixedPrecisionPolicy(manifest: manifest)
    }
}

@Test func mixedPrecisionPolicyRejectsInvalidLayerOverrides() throws {
    var manifest = try loadSampleManifest()
    manifest.quantization.layerOverrides = [.embedding: [0: "int4"]]

    #expect(throws: MixedPrecisionPolicyError.unsupportedLayerOverrideComponent(.embedding)) {
        _ = try MixedPrecisionPolicy(manifest: manifest)
    }

    manifest = try loadSampleManifest()
    manifest.quantization.layerOverrides = [.attentionV: [24: "int4"]]

    #expect(throws: MixedPrecisionPolicyError.invalidLayerOverride(.attentionV, 24)) {
        _ = try MixedPrecisionPolicy(manifest: manifest)
    }
}
