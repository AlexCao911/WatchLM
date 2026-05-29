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
}
