public struct KVTensorLayout: Codable, Equatable, Sendable {
    public var batchSize: Int
    public var kvHeads: Int
    public var contextTokens: Int
    public var headDimension: Int

    public init(
        batchSize: Int,
        kvHeads: Int,
        contextTokens: Int,
        headDimension: Int
    ) {
        self.batchSize = batchSize
        self.kvHeads = kvHeads
        self.contextTokens = contextTokens
        self.headDimension = headDimension
    }

    public var tensorShape: [Int] {
        [batchSize, kvHeads, contextTokens, headDimension]
    }

    public var decodeSliceShape: [Int] {
        [batchSize, kvHeads, 1, headDimension]
    }

    public var scalarCountPerTensor: Int {
        batchSize * kvHeads * contextTokens * headDimension
    }

    public var scalarCountPerToken: Int {
        batchSize * kvHeads * headDimension
    }

    public func byteCount(layerCount: Int, precision: KVCachePrecision) -> Int {
        layerCount * 2 * scalarCountPerTensor * precision.bytesPerScalar
    }

    public func scalarCopyCount(layerCount: Int, movedTokenSlots: Int) -> Int {
        max(0, movedTokenSlots) * scalarCountPerToken * layerCount * 2
    }
}

public enum KVCacheUpdateStrategy: String, Codable, Equatable, Sendable {
    case contiguousSliding
    case slotRing
}
