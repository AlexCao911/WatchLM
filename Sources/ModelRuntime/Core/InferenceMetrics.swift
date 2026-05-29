public struct InferenceMetrics: Codable, Equatable, Sendable {
    public var prefillSamplingMs: Double
    public var decodeSamplingStepMs: [Double]
    public var kvAppendStepMs: [Double]
    public var kvCacheUpdateStrategy: KVCacheUpdateStrategy?
    public var kvAppendWriteIndices: [Int]
    public var kvAppendMovedTokenSlots: [Int]
    public var kvAppendMovedScalarCounts: [Int]

    public init(
        prefillSamplingMs: Double = 0,
        decodeSamplingStepMs: [Double] = [],
        kvAppendStepMs: [Double] = [],
        kvCacheUpdateStrategy: KVCacheUpdateStrategy? = nil,
        kvAppendWriteIndices: [Int] = [],
        kvAppendMovedTokenSlots: [Int] = [],
        kvAppendMovedScalarCounts: [Int] = []
    ) {
        self.prefillSamplingMs = prefillSamplingMs
        self.decodeSamplingStepMs = decodeSamplingStepMs
        self.kvAppendStepMs = kvAppendStepMs
        self.kvCacheUpdateStrategy = kvCacheUpdateStrategy
        self.kvAppendWriteIndices = kvAppendWriteIndices
        self.kvAppendMovedTokenSlots = kvAppendMovedTokenSlots
        self.kvAppendMovedScalarCounts = kvAppendMovedScalarCounts
    }

    public var totalDecodeSamplingMs: Double {
        decodeSamplingStepMs.reduce(0, +)
    }

    public var totalKVAppendMs: Double {
        kvAppendStepMs.reduce(0, +)
    }

    public var totalKVAppendMovedTokenSlots: Int {
        kvAppendMovedTokenSlots.reduce(0, +)
    }

    public var totalKVAppendMovedScalarCount: Int {
        kvAppendMovedScalarCounts.reduce(0, +)
    }
}
