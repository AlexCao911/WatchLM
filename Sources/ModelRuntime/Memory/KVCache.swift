public enum KVCachePrecision: String, Codable, Equatable, Sendable {
    case int8
    case float16

    var bytesPerScalar: Int {
        switch self {
        case .int8:
            1
        case .float16:
            2
        }
    }
}

public struct KVCacheDescriptor: Codable, Equatable, Sendable {
    public var layers: Int
    public var kvHeads: Int
    public var headDimension: Int
    public var contextTokens: Int
    public var precision: KVCachePrecision

    public init(
        layers: Int,
        kvHeads: Int,
        headDimension: Int,
        contextTokens: Int,
        precision: KVCachePrecision
    ) {
        self.layers = layers
        self.kvHeads = kvHeads
        self.headDimension = headDimension
        self.contextTokens = contextTokens
        self.precision = precision
    }

    public static func miniCPM5(
        contextTokens: Int,
        precision: KVCachePrecision
    ) -> KVCacheDescriptor {
        KVCacheDescriptor(
            layers: 24,
            kvHeads: 2,
            headDimension: 128,
            contextTokens: contextTokens,
            precision: precision
        )
    }

    public var bytesPerToken: Int {
        layers * 2 * kvHeads * headDimension * precision.bytesPerScalar
    }

    public var totalBytes: Int {
        bytesPerToken * contextTokens
    }
}
