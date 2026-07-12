#if canImport(CoreML)
import CoreML

struct CoreMLKVCacheStore {
    let layout: KVTensorLayout
    let layerCount: Int
    let updateStrategy: KVCacheUpdateStrategy
    private(set) var validTokenCount: Int
    private(set) var activeTokenStartIndex: Int
    private(set) var lastAppendWriteIndex: Int = 0
    private(set) var lastAppendMovedTokenCount: Int = 0
    private(set) var lastAppendMovedScalarCount: Int = 0
    private let dataType: MLMultiArrayDataType
    private var keys: [MLMultiArray]
    private var values: [MLMultiArray]
    private var freeSlotStack: [Int]
    private var slotOrder: [Int]

    init(
        prefillOutput: MLFeatureProvider,
        layerCount: Int,
        validTokenCount: Int? = nil,
        updateStrategy: KVCacheUpdateStrategy = .contiguousSliding,
        keyOutputName: (Int) -> String,
        valueOutputName: (Int) -> String
    ) throws {
        guard layerCount > 0 else {
            throw InferenceRuntimeError.predictionFailed(message: "KV cache must have at least one layer.")
        }

        let firstKey = try multiArrayOutput(from: prefillOutput, name: keyOutputName(0))
        let firstValue = try multiArrayOutput(from: prefillOutput, name: valueOutputName(0))
        let layout = try Self.layout(for: firstKey, featureName: keyOutputName(0))
        try Self.validate(firstValue, featureName: valueOutputName(0), matches: layout, dataType: firstKey.dataType)

        var keys = [try Self.copy(firstKey)]
        var values = [try Self.copy(firstValue)]
        keys.reserveCapacity(layerCount)
        values.reserveCapacity(layerCount)

        if layerCount > 1 {
            for layer in 1..<layerCount {
                let key = try multiArrayOutput(from: prefillOutput, name: keyOutputName(layer))
                let value = try multiArrayOutput(from: prefillOutput, name: valueOutputName(layer))
                try Self.validate(key, featureName: keyOutputName(layer), matches: layout, dataType: firstKey.dataType)
                try Self.validate(value, featureName: valueOutputName(layer), matches: layout, dataType: firstKey.dataType)
                keys.append(try Self.copy(key))
                values.append(try Self.copy(value))
            }
        }

        self.layout = layout
        self.layerCount = layerCount
        let initialValidTokenCount = validTokenCount ?? layout.contextTokens
        guard (0...layout.contextTokens).contains(initialValidTokenCount) else {
            throw InferenceRuntimeError.predictionFailed(message: "validTokenCount must fit inside the KV cache context window.")
        }
        self.validTokenCount = initialValidTokenCount
        let initialActiveTokenStartIndex = layout.contextTokens - initialValidTokenCount
        activeTokenStartIndex = initialActiveTokenStartIndex
        self.updateStrategy = updateStrategy
        freeSlotStack = Array(0..<initialActiveTokenStartIndex)
        slotOrder = initialValidTokenCount > 0 ? Array(initialActiveTokenStartIndex..<layout.contextTokens) : []
        dataType = firstKey.dataType
        self.keys = keys
        self.values = values
    }

    func key(forLayer layer: Int) -> MLMultiArray {
        keys[layer]
    }

    func value(forLayer layer: Int) -> MLMultiArray {
        values[layer]
    }

    mutating func appendDecodeOutputs(
        output: MLFeatureProvider,
        keyOutputName: (Int) -> String,
        valueOutputName: (Int) -> String
    ) throws {
        let plan = appendPlan()
        for layer in 0..<layerCount {
            let newKey = try multiArrayOutput(from: output, name: keyOutputName(layer))
            let newValue = try multiArrayOutput(from: output, name: valueOutputName(layer))
            try Self.validateDecodeSlice(newKey, featureName: keyOutputName(layer), shape: layout.decodeSliceShape, storeDataType: dataType)
            try Self.validateDecodeSlice(newValue, featureName: valueOutputName(layer), shape: layout.decodeSliceShape, storeDataType: dataType)
            try append(newSlice: newKey, to: keys[layer], plan: plan)
            try append(newSlice: newValue, to: values[layer], plan: plan)
        }

        lastAppendWriteIndex = plan.writeIndex
        lastAppendMovedTokenCount = plan.copiedTokenCount
        lastAppendMovedScalarCount = layout.scalarCopyCount(
            layerCount: layerCount,
            movedTokenSlots: plan.copiedTokenCount
        )
        commitAppendPlan(plan)
    }

    private func append(newSlice: MLMultiArray, to past: MLMultiArray, plan: AppendPlan) throws {
        let contextTokens = layout.contextTokens
        guard contextTokens > 0 else {
            throw InferenceRuntimeError.predictionFailed(message: "KV cache must have at least one context token.")
        }

        if plan.copiedTokenCount > 0 {
            for batch in 0..<layout.batchSize {
                for head in 0..<layout.kvHeads {
                    for tokenOffset in 0..<plan.copiedTokenCount {
                        for dimension in 0..<layout.headDimension {
                            past[linearIndex(in: past, [batch, head, plan.destinationStart + tokenOffset, dimension])] =
                                past[linearIndex(in: past, [batch, head, plan.sourceStart + tokenOffset, dimension])]
                        }
                    }
                }
            }
        }

        for batch in 0..<layout.batchSize {
            for head in 0..<layout.kvHeads {
                for dimension in 0..<layout.headDimension {
                    past[linearIndex(in: past, [batch, head, plan.writeIndex, dimension])] =
                        newSlice[linearIndex(in: newSlice, [batch, head, 0, dimension])]
                }
            }
        }
    }

    private func appendPlan() -> AppendPlan {
        switch updateStrategy {
        case .contiguousSliding:
            return contiguousSlidingAppendPlan()
        case .slotRing:
            return slotRingAppendPlan()
        }
    }

    private func contiguousSlidingAppendPlan() -> AppendPlan {
        if activeTokenStartIndex > 0 {
            let destinationStart = activeTokenStartIndex - 1
            return AppendPlan(
                sourceStart: activeTokenStartIndex,
                destinationStart: destinationStart,
                copiedTokenCount: validTokenCount,
                writeIndex: destinationStart + validTokenCount
            )
        }

        return AppendPlan(
            sourceStart: 1,
            destinationStart: 0,
            copiedTokenCount: max(0, layout.contextTokens - 1),
            writeIndex: layout.contextTokens - 1
        )
    }

    private func slotRingAppendPlan() -> AppendPlan {
        let writeIndex = freeSlotStack.last ?? slotOrder.first ?? 0
        return AppendPlan(
            sourceStart: 0,
            destinationStart: 0,
            copiedTokenCount: 0,
            writeIndex: writeIndex
        )
    }

    private mutating func commitAppendPlan(_ plan: AppendPlan) {
        switch updateStrategy {
        case .contiguousSliding:
            if activeTokenStartIndex > 0 {
                activeTokenStartIndex -= 1
                validTokenCount += 1
            }
        case .slotRing:
            if !freeSlotStack.isEmpty {
                _ = freeSlotStack.removeLast()
                validTokenCount += 1
                activeTokenStartIndex = freeSlotStack.count
            } else if !slotOrder.isEmpty {
                _ = slotOrder.removeFirst()
            }
            slotOrder.append(plan.writeIndex)
        }
    }

    private static func layout(for array: MLMultiArray, featureName: String) throws -> KVTensorLayout {
        let shape = array.shape.map(\.intValue)
        guard shape.count == 4 else {
            throw InferenceRuntimeError.predictionFailed(message: "\(featureName) must use [batch, heads, tokens, head_dim] shape.")
        }

        guard shape.allSatisfy({ $0 > 0 }) else {
            throw InferenceRuntimeError.predictionFailed(message: "\(featureName) KV shape must be positive.")
        }

        return KVTensorLayout(
            batchSize: shape[0],
            kvHeads: shape[1],
            contextTokens: shape[2],
            headDimension: shape[3]
        )
    }

    private static func validate(
        _ array: MLMultiArray,
        featureName: String,
        matches layout: KVTensorLayout,
        dataType: MLMultiArrayDataType
    ) throws {
        try validate(array, featureName: featureName, shape: layout.tensorShape, dataType: dataType)
    }

    private static func validate(
        _ array: MLMultiArray,
        featureName: String,
        shape expectedShape: [Int],
        dataType expectedDataType: MLMultiArrayDataType
    ) throws {
        let shape = array.shape.map(\.intValue)
        guard shape == expectedShape else {
            throw InferenceRuntimeError.predictionFailed(message: "\(featureName) shape \(shape) does not match expected KV shape \(expectedShape).")
        }

        guard array.dataType == expectedDataType else {
            throw InferenceRuntimeError.predictionFailed(message: "\(featureName) data type does not match the KV cache store.")
        }
    }

    private static func validateDecodeSlice(
        _ array: MLMultiArray,
        featureName: String,
        shape expectedShape: [Int],
        storeDataType: MLMultiArrayDataType
    ) throws {
        let shape = array.shape.map(\.intValue)
        guard shape == expectedShape else {
            throw InferenceRuntimeError.predictionFailed(message: "\(featureName) shape \(shape) does not match expected KV shape \(expectedShape).")
        }

        guard array.dataType == storeDataType || (isFloatKVDataType(array.dataType) && isFloatKVDataType(storeDataType)) else {
            throw InferenceRuntimeError.predictionFailed(message: "\(featureName) data type does not match the KV cache store.")
        }
    }

    private static func isFloatKVDataType(_ dataType: MLMultiArrayDataType) -> Bool {
        dataType == .float16 || dataType == .float32
    }

    private static func copy(_ source: MLMultiArray) throws -> MLMultiArray {
        let destination = try MLMultiArray(shape: source.shape, dataType: source.dataType)
        for index in 0..<source.count {
            destination[index] = source[index]
        }
        return destination
    }

    private func linearIndex(in array: MLMultiArray, _ indexes: [Int]) -> Int {
        zip(indexes, array.strides.map(\.intValue)).reduce(0) { partial, pair in
            partial + pair.0 * pair.1
        }
    }

    private struct AppendPlan {
        var sourceStart: Int
        var destinationStart: Int
        var copiedTokenCount: Int
        var writeIndex: Int
    }
}
#endif
