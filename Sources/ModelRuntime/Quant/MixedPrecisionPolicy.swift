public enum QuantizedPrecision: String, Codable, Equatable, Sendable {
    case fp16
    case int8
    case int4

    fileprivate func raisedToAtLeastInt8() -> QuantizedPrecision {
        switch self {
        case .fp16, .int8:
            self
        case .int4:
            .int8
        }
    }
}

public enum MixedPrecisionComponent: String, Codable, Equatable, Sendable {
    case embedding
    case lmHead
    case norms
    case attentionQKO
    case attentionV
    case ffn
}

public enum MixedPrecisionPolicyError: Error, Equatable, Sendable {
    case unsupportedStrategy(String)
    case unsupportedPrecision(String)
    case unsupportedKVCachePrecision(String)
    case structuralReductionEnabled
    case invalidLayerCount(Int)
    case invalidProtectedLayerCount(Int)
}

public struct MixedPrecisionPolicy: Codable, Equatable, Sendable {
    public var layerCount: Int
    public var protectedEdgeLayerCount: Int
    public var embedding: QuantizedPrecision
    public var lmHead: QuantizedPrecision
    public var norms: QuantizedPrecision
    public var attentionQKO: QuantizedPrecision
    public var attentionV: QuantizedPrecision
    public var ffn: QuantizedPrecision
    public var kvCache: QuantizedPrecision

    public init(
        manifest: ModelManifest,
        protectedEdgeLayerCount: Int = 2
    ) throws {
        try self.init(
            quantization: manifest.quantization,
            layerCount: manifest.architecture.layers,
            protectedEdgeLayerCount: protectedEdgeLayerCount
        )
    }

    public init(
        quantization: QuantizationInfo,
        layerCount: Int,
        protectedEdgeLayerCount: Int = 2
    ) throws {
        guard quantization.strategy == "mixed-precision-fidelity-first" else {
            throw MixedPrecisionPolicyError.unsupportedStrategy(quantization.strategy)
        }

        guard !quantization.structuralReduction else {
            throw MixedPrecisionPolicyError.structuralReductionEnabled
        }

        guard layerCount > 0 else {
            throw MixedPrecisionPolicyError.invalidLayerCount(layerCount)
        }

        guard protectedEdgeLayerCount >= 0 else {
            throw MixedPrecisionPolicyError.invalidProtectedLayerCount(protectedEdgeLayerCount)
        }

        let kvCache = try Self.parsePrecision(quantization.kvCache)
        guard kvCache == .int8 else {
            throw MixedPrecisionPolicyError.unsupportedKVCachePrecision(quantization.kvCache)
        }

        self.layerCount = layerCount
        self.protectedEdgeLayerCount = min(protectedEdgeLayerCount, layerCount)
        embedding = try Self.parsePrecision(quantization.weights.embedding)
        lmHead = try Self.parsePrecision(quantization.weights.lmHead)
        norms = try Self.parsePrecision(quantization.weights.norms)
        attentionQKO = try Self.parsePrecision(quantization.weights.attentionQKO)
        attentionV = try Self.parsePrecision(quantization.weights.attentionV)
        ffn = try Self.parsePrecision(quantization.weights.ffn)
        self.kvCache = kvCache
    }

    public func shouldProtectTransformerLayer(_ layer: Int) -> Bool {
        guard (0..<layerCount).contains(layer) else {
            return false
        }

        return layer < protectedEdgeLayerCount || layer >= layerCount - protectedEdgeLayerCount
    }

    public func precision(
        for component: MixedPrecisionComponent,
        layer: Int? = nil
    ) -> QuantizedPrecision {
        let basePrecision: QuantizedPrecision
        switch component {
        case .embedding:
            basePrecision = embedding
        case .lmHead:
            basePrecision = lmHead
        case .norms:
            basePrecision = norms
        case .attentionQKO:
            basePrecision = attentionQKO
        case .attentionV:
            basePrecision = attentionV
        case .ffn:
            basePrecision = ffn
        }

        guard let layer, isTransformerComponent(component), shouldProtectTransformerLayer(layer) else {
            return basePrecision
        }

        return basePrecision.raisedToAtLeastInt8()
    }

    public var kvCacheDescriptorPrecision: KVCachePrecision {
        switch kvCache {
        case .int8:
            .int8
        case .fp16, .int4:
            .float16
        }
    }

    private static func parsePrecision(_ rawValue: String) throws -> QuantizedPrecision {
        guard let precision = QuantizedPrecision(rawValue: rawValue) else {
            throw MixedPrecisionPolicyError.unsupportedPrecision(rawValue)
        }
        return precision
    }

    private func isTransformerComponent(_ component: MixedPrecisionComponent) -> Bool {
        switch component {
        case .attentionQKO, .attentionV, .ffn:
            true
        case .embedding, .lmHead, .norms:
            false
        }
    }
}
