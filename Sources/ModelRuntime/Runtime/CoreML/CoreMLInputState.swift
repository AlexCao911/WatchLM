#if canImport(CoreML)
import CoreML
import Foundation

struct CoreMLMiniCPMInputState {
    let inputIDs: MLMultiArray
    let positionIDs: MLMultiArray
    let causalMask: MLMultiArray
    private(set) var tokenMask: [Bool]
    private var nextPositionIDValue: Int

    init(
        tokenIDs: [Int32],
        capacity: Int,
        padTokenID: Int32 = MiniCPMSpecialTokens.padTokenID
    ) throws {
        guard capacity > 0 else {
            throw InferenceRuntimeError.invalidInput(message: "maxPromptTokens must be positive.")
        }

        let suffix = Array(tokenIDs.suffix(capacity))
        let paddingCount = capacity - suffix.count
        var paddedTokens = Array(repeating: padTokenID, count: paddingCount)
        paddedTokens.append(contentsOf: suffix)

        var tokenMask = Array(repeating: false, count: paddingCount)
        tokenMask.append(contentsOf: Array(repeating: true, count: suffix.count))

        inputIDs = try MLMultiArray(shape: [1, NSNumber(value: capacity)], dataType: .int32)
        positionIDs = try MLMultiArray(shape: [1, NSNumber(value: capacity)], dataType: .int32)
        causalMask = try MLMultiArray(
            shape: [1, 1, NSNumber(value: capacity), NSNumber(value: capacity)],
            dataType: .float16
        )
        self.tokenMask = tokenMask
        nextPositionIDValue = suffix.count

        var runningPosition = -1
        for index in 0..<capacity {
            inputIDs[[0, index] as [NSNumber]] = NSNumber(value: paddedTokens[index])
            if tokenMask[index] {
                runningPosition += 1
            }
            positionIDs[[0, index] as [NSNumber]] = NSNumber(value: max(runningPosition, 0))
        }

        for queryIndex in 0..<capacity {
            for keyIndex in 0..<capacity {
                let allowed = keyIndex <= queryIndex && tokenMask[keyIndex]
                causalMask[[0, 0, queryIndex, keyIndex] as [NSNumber]] = NSNumber(value: allowed ? 0 : -65504)
            }
        }
    }

    var realTokenCount: Int {
        tokenMask.filter { $0 }.count
    }

    var decodePositionID: MLMultiArray {
        let array = try! MLMultiArray(shape: [1, 1], dataType: .int32)
        array[0] = NSNumber(value: nextPositionIDValue)
        return array
    }

    var decodeCausalMask: MLMultiArray {
        let contextTokens = tokenMask.count
        let array = try! MLMultiArray(
            shape: [1, 1, 1, NSNumber(value: contextTokens + 1)],
            dataType: .float16
        )

        for keyIndex in 0..<contextTokens {
            array[[0, 0, 0, keyIndex] as [NSNumber]] = NSNumber(value: tokenMask[keyIndex] ? 0 : -65504)
        }
        array[[0, 0, 0, contextTokens] as [NSNumber]] = NSNumber(value: 0)
        return array
    }

    mutating func appendGeneratedToken() {
        guard !tokenMask.isEmpty else {
            return
        }

        tokenMask.removeFirst()
        tokenMask.append(true)
        nextPositionIDValue += 1
    }

    mutating func appendGeneratedToken(atPastKVSlot slotIndex: Int) throws {
        guard tokenMask.indices.contains(slotIndex) else {
            throw InferenceRuntimeError.predictionFailed(message: "Generated token KV slot \(slotIndex) is outside the decode context window.")
        }

        tokenMask[slotIndex] = true
        nextPositionIDValue += 1
    }
}

func paddedPromptArray(tokenIDs: [Int32], capacity: Int) throws -> MLMultiArray {
    guard capacity > 0 else {
        throw InferenceRuntimeError.invalidInput(message: "maxPromptTokens must be positive.")
    }

    let array = try MLMultiArray(shape: [NSNumber(value: capacity)], dataType: .double)
    let suffix = tokenIDs.suffix(capacity)
    for index in 0..<capacity {
        let tokenIndex = suffix.index(suffix.startIndex, offsetBy: index, limitedBy: suffix.index(before: suffix.endIndex))
        let tokenID = tokenIndex.map { suffix[$0] } ?? 0
        array[index] = NSNumber(value: Double(tokenID))
    }
    return array
}

func scalarArray(_ value: Double) throws -> MLMultiArray {
    let array = try MLMultiArray(shape: [1], dataType: .double)
    array[0] = NSNumber(value: value)
    return array
}

func tokenIDArray(_ value: Int32) throws -> MLMultiArray {
    let array = try MLMultiArray(shape: [1, 1], dataType: .int32)
    array[0] = NSNumber(value: value)
    return array
}
#endif
