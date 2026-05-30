#if canImport(CoreML)
import CoreML
import Foundation

public enum CoreMLPrefillDecodeGraphInterface: Equatable, Sendable {
    case tokenAndSingleKV
    case logitsAndLayeredKV(layerCount: Int, kvHeads: Int, headDimension: Int)
    case statefulKV(layerCount: Int, kvHeads: Int, headDimension: Int)
}

public struct CoreMLPrefillDecodeBundle: Sendable {
    public var prefillModelURL: URL
    public var decodeModelURL: URL
    public var maxPromptTokens: Int
    public var graphInterface: CoreMLPrefillDecodeGraphInterface
    public var prefillInputName: String
    public var prefillPositionInputName: String
    public var prefillCausalMaskInputName: String
    public var prefillNextTokenOutputName: String
    public var prefillLogitsOutputName: String
    public var prefillKVCacheOutputName: String
    public var prefillKeyOutputNamePrefix: String
    public var prefillValueOutputNamePrefix: String
    public var decodeTokenInputName: String
    public var decodePositionInputName: String
    public var decodeCausalMaskInputName: String
    public var decodeKVCacheInputName: String
    public var decodeNextTokenOutputName: String
    public var decodeLogitsOutputName: String
    public var decodeKVCacheOutputName: String
    public var decodePastKeyInputNamePrefix: String
    public var decodePastValueInputNamePrefix: String
    public var decodeNewKeyOutputNamePrefix: String
    public var decodeNewValueOutputNamePrefix: String
    public var logitsProcessor: LogitsProcessor
    public var samplingStrategy: TokenSamplingStrategy
    public var kvCacheUpdateStrategy: KVCacheUpdateStrategy

    public init(
        prefillModelURL: URL,
        decodeModelURL: URL,
        maxPromptTokens: Int,
        graphInterface: CoreMLPrefillDecodeGraphInterface = .tokenAndSingleKV,
        prefillInputName: String = "input_ids",
        prefillPositionInputName: String = "position_ids",
        prefillCausalMaskInputName: String = "causal_mask",
        prefillNextTokenOutputName: String = "next_token",
        prefillLogitsOutputName: String = "logits",
        prefillKVCacheOutputName: String = "kv_cache",
        prefillKeyOutputNamePrefix: String = "present_key_",
        prefillValueOutputNamePrefix: String = "present_value_",
        decodeTokenInputName: String = "token",
        decodePositionInputName: String = "position_id",
        decodeCausalMaskInputName: String = "causal_mask",
        decodeKVCacheInputName: String = "kv_cache",
        decodeNextTokenOutputName: String = "next_token",
        decodeLogitsOutputName: String = "logits",
        decodeKVCacheOutputName: String = "updated_kv_cache",
        decodePastKeyInputNamePrefix: String = "past_key_",
        decodePastValueInputNamePrefix: String = "past_value_",
        decodeNewKeyOutputNamePrefix: String = "new_key_",
        decodeNewValueOutputNamePrefix: String = "new_value_",
        logitsProcessor: LogitsProcessor = LogitsProcessor(),
        samplingStrategy: TokenSamplingStrategy = .greedy,
        kvCacheUpdateStrategy: KVCacheUpdateStrategy = .slotRing
    ) {
        self.prefillModelURL = prefillModelURL
        self.decodeModelURL = decodeModelURL
        self.maxPromptTokens = maxPromptTokens
        self.graphInterface = graphInterface
        self.prefillInputName = prefillInputName
        self.prefillPositionInputName = prefillPositionInputName
        self.prefillCausalMaskInputName = prefillCausalMaskInputName
        self.prefillNextTokenOutputName = prefillNextTokenOutputName
        self.prefillLogitsOutputName = prefillLogitsOutputName
        self.prefillKVCacheOutputName = prefillKVCacheOutputName
        self.prefillKeyOutputNamePrefix = prefillKeyOutputNamePrefix
        self.prefillValueOutputNamePrefix = prefillValueOutputNamePrefix
        self.decodeTokenInputName = decodeTokenInputName
        self.decodePositionInputName = decodePositionInputName
        self.decodeCausalMaskInputName = decodeCausalMaskInputName
        self.decodeKVCacheInputName = decodeKVCacheInputName
        self.decodeNextTokenOutputName = decodeNextTokenOutputName
        self.decodeLogitsOutputName = decodeLogitsOutputName
        self.decodeKVCacheOutputName = decodeKVCacheOutputName
        self.decodePastKeyInputNamePrefix = decodePastKeyInputNamePrefix
        self.decodePastValueInputNamePrefix = decodePastValueInputNamePrefix
        self.decodeNewKeyOutputNamePrefix = decodeNewKeyOutputNamePrefix
        self.decodeNewValueOutputNamePrefix = decodeNewValueOutputNamePrefix
        self.logitsProcessor = logitsProcessor
        self.samplingStrategy = samplingStrategy
        self.kvCacheUpdateStrategy = kvCacheUpdateStrategy
    }

    public init(
        prefillModelURL: URL,
        decodeModelURL: URL,
        maxPromptTokens: Int,
        graphSchema: RuntimeGraphSchema,
        logitsProcessor: LogitsProcessor = LogitsProcessor(),
        samplingStrategy: TokenSamplingStrategy = .greedy,
        kvCacheUpdateStrategy: KVCacheUpdateStrategy = .slotRing
    ) throws {
        let graphInterface: CoreMLPrefillDecodeGraphInterface
        switch graphSchema.interface {
        case "logits-layered-kv":
            graphInterface = .logitsAndLayeredKV(
                layerCount: graphSchema.layerCount,
                kvHeads: graphSchema.kvHeads,
                headDimension: graphSchema.headDimension
            )
        case "stateful-kv":
            graphInterface = .statefulKV(
                layerCount: graphSchema.layerCount,
                kvHeads: graphSchema.kvHeads,
                headDimension: graphSchema.headDimension
            )
        default:
            throw InferenceRuntimeError.invalidInput(message: "Unsupported Core ML graph interface \(graphSchema.interface).")
        }

        self.init(
            prefillModelURL: prefillModelURL,
            decodeModelURL: decodeModelURL,
            maxPromptTokens: maxPromptTokens,
            graphInterface: graphInterface,
            prefillInputName: graphSchema.prefill.inputIDs,
            prefillPositionInputName: graphSchema.prefill.positionIDs,
            prefillCausalMaskInputName: graphSchema.prefill.causalMask,
            prefillLogitsOutputName: graphSchema.prefill.logits,
            prefillKeyOutputNamePrefix: graphSchema.prefill.keyPrefix,
            prefillValueOutputNamePrefix: graphSchema.prefill.valuePrefix,
            decodeTokenInputName: graphSchema.decode.tokenID,
            decodePositionInputName: graphSchema.decode.positionID,
            decodeCausalMaskInputName: graphSchema.decode.causalMask,
            decodeLogitsOutputName: graphSchema.decode.logits,
            decodePastKeyInputNamePrefix: graphSchema.decode.pastKeyPrefix,
            decodePastValueInputNamePrefix: graphSchema.decode.pastValuePrefix,
            decodeNewKeyOutputNamePrefix: graphSchema.decode.newKeyPrefix,
            decodeNewValueOutputNamePrefix: graphSchema.decode.newValuePrefix,
            logitsProcessor: logitsProcessor,
            samplingStrategy: samplingStrategy,
            kvCacheUpdateStrategy: kvCacheUpdateStrategy
        )
    }

    public var requiresSharedStatefulModel: Bool {
        if case .statefulKV = graphInterface {
            return true
        }
        return false
    }

    public static func miniCPMExplicitKV(
        prefillModelURL: URL,
        decodeModelURL: URL,
        maxPromptTokens: Int
    ) -> CoreMLPrefillDecodeBundle {
        CoreMLPrefillDecodeBundle(
            prefillModelURL: prefillModelURL,
            decodeModelURL: decodeModelURL,
            maxPromptTokens: maxPromptTokens,
            graphInterface: .logitsAndLayeredKV(layerCount: 24, kvHeads: 2, headDimension: 128),
            decodeTokenInputName: "token_id"
        )
    }

    public func prefillKeyOutputName(forLayer layer: Int) -> String {
        "\(prefillKeyOutputNamePrefix)\(layer)"
    }

    public func prefillValueOutputName(forLayer layer: Int) -> String {
        "\(prefillValueOutputNamePrefix)\(layer)"
    }

    public func decodePastKeyInputName(forLayer layer: Int) -> String {
        "\(decodePastKeyInputNamePrefix)\(layer)"
    }

    public func decodePastValueInputName(forLayer layer: Int) -> String {
        "\(decodePastValueInputNamePrefix)\(layer)"
    }

    public func decodeNewKeyOutputName(forLayer layer: Int) -> String {
        "\(decodeNewKeyOutputNamePrefix)\(layer)"
    }

    public func decodeNewValueOutputName(forLayer layer: Int) -> String {
        "\(decodeNewValueOutputNamePrefix)\(layer)"
    }

    public func validateGraphIOContract(
        prefillInputNames: [String],
        prefillOutputNames: [String],
        decodeInputNames: [String],
        decodeOutputNames: [String]
    ) throws {
        let missingGroups = graphIOMissingGroups(
            prefillInputNames: Set(prefillInputNames),
            prefillOutputNames: Set(prefillOutputNames),
            decodeInputNames: Set(decodeInputNames),
            decodeOutputNames: Set(decodeOutputNames)
        )

        guard missingGroups.isEmpty else {
            throw InferenceRuntimeError.invalidInput(
                message: "Core ML graph IO contract mismatch: missing \(missingGroups.joined(separator: "; missing "))."
            )
        }
    }

    public func validateGraphIOContract(
        prefillInputShapes: [String: [Int]],
        prefillOutputShapes: [String: [Int]],
        decodeInputShapes: [String: [Int]],
        decodeOutputShapes: [String: [Int]]
    ) throws {
        try validateGraphIOContract(
            prefillInputNames: Array(prefillInputShapes.keys),
            prefillOutputNames: Array(prefillOutputShapes.keys),
            decodeInputNames: Array(decodeInputShapes.keys),
            decodeOutputNames: Array(decodeOutputShapes.keys)
        )

        let mismatchGroups = graphIOShapeMismatchGroups(
            prefillInputShapes: prefillInputShapes,
            prefillOutputShapes: prefillOutputShapes,
            decodeInputShapes: decodeInputShapes,
            decodeOutputShapes: decodeOutputShapes
        )

        guard mismatchGroups.isEmpty else {
            throw InferenceRuntimeError.invalidInput(
                message: "Core ML graph IO shape mismatch: \(mismatchGroups.joined(separator: "; "))."
            )
        }
    }

    public func validateGraphIOContract(
        prefillInputDataTypes: [String: MLMultiArrayDataType],
        prefillOutputDataTypes: [String: MLMultiArrayDataType],
        decodeInputDataTypes: [String: MLMultiArrayDataType],
        decodeOutputDataTypes: [String: MLMultiArrayDataType]
    ) throws {
        try validateGraphIOContract(
            prefillInputNames: Array(prefillInputDataTypes.keys),
            prefillOutputNames: Array(prefillOutputDataTypes.keys),
            decodeInputNames: Array(decodeInputDataTypes.keys),
            decodeOutputNames: Array(decodeOutputDataTypes.keys)
        )

        let mismatchGroups = graphIODataTypeMismatchGroups(
            prefillInputDataTypes: prefillInputDataTypes,
            prefillOutputDataTypes: prefillOutputDataTypes,
            decodeInputDataTypes: decodeInputDataTypes,
            decodeOutputDataTypes: decodeOutputDataTypes
        )

        guard mismatchGroups.isEmpty else {
            throw InferenceRuntimeError.invalidInput(
                message: "Core ML graph IO dtype mismatch: \(mismatchGroups.joined(separator: "; "))."
            )
        }
    }

    public func validateModelDescriptions(
        prefill: MLModelDescription,
        decode: MLModelDescription
    ) throws {
        try validateGraphIOContract(
            prefillInputShapes: multiArrayShapes(from: prefill.inputDescriptionsByName),
            prefillOutputShapes: multiArrayShapes(from: prefill.outputDescriptionsByName),
            decodeInputShapes: multiArrayShapes(from: decode.inputDescriptionsByName),
            decodeOutputShapes: multiArrayShapes(from: decode.outputDescriptionsByName)
        )
        try validateGraphIOContract(
            prefillInputDataTypes: multiArrayDataTypes(from: prefill.inputDescriptionsByName),
            prefillOutputDataTypes: multiArrayDataTypes(from: prefill.outputDescriptionsByName),
            decodeInputDataTypes: multiArrayDataTypes(from: decode.inputDescriptionsByName),
            decodeOutputDataTypes: multiArrayDataTypes(from: decode.outputDescriptionsByName)
        )
    }

    private func graphIOMissingGroups(
        prefillInputNames: Set<String>,
        prefillOutputNames: Set<String>,
        decodeInputNames: Set<String>,
        decodeOutputNames: Set<String>
    ) -> [String] {
        let required = requiredGraphIONames()
        return [
            missingGroup("prefill inputs", required.prefillInputs, available: prefillInputNames),
            missingGroup("prefill outputs", required.prefillOutputs, available: prefillOutputNames),
            missingGroup("decode inputs", required.decodeInputs, available: decodeInputNames),
            missingGroup("decode outputs", required.decodeOutputs, available: decodeOutputNames)
        ].compactMap { $0 }
    }

    private func requiredGraphIONames() -> (
        prefillInputs: [String],
        prefillOutputs: [String],
        decodeInputs: [String],
        decodeOutputs: [String]
    ) {
        switch graphInterface {
        case .tokenAndSingleKV:
            return (
                prefillInputs: [prefillInputName],
                prefillOutputs: [prefillNextTokenOutputName, prefillKVCacheOutputName],
                decodeInputs: [decodeTokenInputName, decodeKVCacheInputName],
                decodeOutputs: [decodeNextTokenOutputName, decodeKVCacheOutputName]
            )
        case .statefulKV:
            return (
                prefillInputs: [
                    prefillInputName,
                    prefillPositionInputName,
                    prefillCausalMaskInputName
                ],
                prefillOutputs: [prefillLogitsOutputName],
                decodeInputs: [
                    decodeTokenInputName,
                    decodePositionInputName,
                    decodeCausalMaskInputName
                ],
                decodeOutputs: [decodeLogitsOutputName]
            )
        case .logitsAndLayeredKV(let layerCount, _, _):
            var prefillOutputs = [prefillLogitsOutputName]
            var decodeInputs = [
                decodeTokenInputName,
                decodePositionInputName,
                decodeCausalMaskInputName
            ]
            var decodeOutputs = [decodeLogitsOutputName]

            for layer in 0..<layerCount {
                prefillOutputs.append(prefillKeyOutputName(forLayer: layer))
                prefillOutputs.append(prefillValueOutputName(forLayer: layer))
                decodeInputs.append(decodePastKeyInputName(forLayer: layer))
                decodeInputs.append(decodePastValueInputName(forLayer: layer))
                decodeOutputs.append(decodeNewKeyOutputName(forLayer: layer))
                decodeOutputs.append(decodeNewValueOutputName(forLayer: layer))
            }

            return (
                prefillInputs: [
                    prefillInputName,
                    prefillPositionInputName,
                    prefillCausalMaskInputName
                ],
                prefillOutputs: prefillOutputs,
                decodeInputs: decodeInputs,
                decodeOutputs: decodeOutputs
            )
        }
    }

    private func graphIOShapeMismatchGroups(
        prefillInputShapes: [String: [Int]],
        prefillOutputShapes: [String: [Int]],
        decodeInputShapes: [String: [Int]],
        decodeOutputShapes: [String: [Int]]
    ) -> [String] {
        let expected = requiredGraphIOShapeSpecs()
        return [
            shapeMismatchGroup("prefill inputs", expected.prefillInputs, actual: prefillInputShapes),
            shapeMismatchGroup("prefill outputs", expected.prefillOutputs, actual: prefillOutputShapes),
            shapeMismatchGroup("decode inputs", expected.decodeInputs, actual: decodeInputShapes),
            shapeMismatchGroup("decode outputs", expected.decodeOutputs, actual: decodeOutputShapes)
        ].compactMap { $0 }
    }

    private func requiredGraphIOShapeSpecs() -> (
        prefillInputs: [String: [Int?]],
        prefillOutputs: [String: [Int?]],
        decodeInputs: [String: [Int?]],
        decodeOutputs: [String: [Int?]]
    ) {
        switch graphInterface {
        case .tokenAndSingleKV:
            return (
                prefillInputs: [prefillInputName: [maxPromptTokens]],
                prefillOutputs: [
                    prefillNextTokenOutputName: [1],
                    prefillKVCacheOutputName: [1]
                ],
                decodeInputs: [
                    decodeTokenInputName: [1],
                    decodeKVCacheInputName: [1]
                ],
                decodeOutputs: [
                    decodeNextTokenOutputName: [1],
                    decodeKVCacheOutputName: [1]
                ]
            )
        case .statefulKV:
            return (
                prefillInputs: [
                    prefillInputName: [1, nil],
                    prefillPositionInputName: [1, nil],
                    prefillCausalMaskInputName: [1, 1, nil, nil]
                ],
                prefillOutputs: [
                    prefillLogitsOutputName: [1, nil]
                ],
                decodeInputs: [
                    decodeTokenInputName: [1, nil],
                    decodePositionInputName: [1, nil],
                    decodeCausalMaskInputName: [1, 1, nil, nil]
                ],
                decodeOutputs: [
                    decodeLogitsOutputName: [1, nil]
                ]
            )
        case .logitsAndLayeredKV(let layerCount, let kvHeads, let headDimension):
            let kvShape: [Int?] = [1, kvHeads, maxPromptTokens, headDimension]
            let decodeSliceShape: [Int?] = [1, kvHeads, 1, headDimension]
            var prefillOutputs: [String: [Int?]] = [
                prefillLogitsOutputName: [1, nil]
            ]
            var decodeInputs: [String: [Int?]] = [
                decodeTokenInputName: [1, 1],
                decodePositionInputName: [1, 1],
                decodeCausalMaskInputName: [1, 1, 1, maxPromptTokens + 1]
            ]
            var decodeOutputs: [String: [Int?]] = [
                decodeLogitsOutputName: [1, nil]
            ]

            for layer in 0..<layerCount {
                prefillOutputs[prefillKeyOutputName(forLayer: layer)] = kvShape
                prefillOutputs[prefillValueOutputName(forLayer: layer)] = kvShape
                decodeInputs[decodePastKeyInputName(forLayer: layer)] = kvShape
                decodeInputs[decodePastValueInputName(forLayer: layer)] = kvShape
                decodeOutputs[decodeNewKeyOutputName(forLayer: layer)] = decodeSliceShape
                decodeOutputs[decodeNewValueOutputName(forLayer: layer)] = decodeSliceShape
            }

            return (
                prefillInputs: [
                    prefillInputName: [1, maxPromptTokens],
                    prefillPositionInputName: [1, maxPromptTokens],
                    prefillCausalMaskInputName: [1, 1, maxPromptTokens, maxPromptTokens]
                ],
                prefillOutputs: prefillOutputs,
                decodeInputs: decodeInputs,
                decodeOutputs: decodeOutputs
            )
        }
    }

    private func graphIODataTypeMismatchGroups(
        prefillInputDataTypes: [String: MLMultiArrayDataType],
        prefillOutputDataTypes: [String: MLMultiArrayDataType],
        decodeInputDataTypes: [String: MLMultiArrayDataType],
        decodeOutputDataTypes: [String: MLMultiArrayDataType]
    ) -> [String] {
        let expected = requiredGraphIODataTypeSpecs()
        return [
            dataTypeMismatchGroup("prefill inputs", expected.prefillInputs, actual: prefillInputDataTypes),
            dataTypeMismatchGroup("prefill outputs", expected.prefillOutputs, actual: prefillOutputDataTypes),
            dataTypeMismatchGroup("decode inputs", expected.decodeInputs, actual: decodeInputDataTypes),
            dataTypeMismatchGroup("decode outputs", expected.decodeOutputs, actual: decodeOutputDataTypes)
        ].compactMap { $0 }
    }

    private func requiredGraphIODataTypeSpecs() -> (
        prefillInputs: [String: [MLMultiArrayDataType]],
        prefillOutputs: [String: [MLMultiArrayDataType]],
        decodeInputs: [String: [MLMultiArrayDataType]],
        decodeOutputs: [String: [MLMultiArrayDataType]]
    ) {
        switch graphInterface {
        case .tokenAndSingleKV:
            return (
                prefillInputs: [prefillInputName: [.double]],
                prefillOutputs: [
                    prefillNextTokenOutputName: [.double],
                    prefillKVCacheOutputName: [.double]
                ],
                decodeInputs: [
                    decodeTokenInputName: [.double],
                    decodeKVCacheInputName: [.double]
                ],
                decodeOutputs: [
                    decodeNextTokenOutputName: [.double],
                    decodeKVCacheOutputName: [.double]
                ]
            )
        case .statefulKV:
            let floatingLogits: [MLMultiArrayDataType] = [.float16, .float32, .double]
            return (
                prefillInputs: [
                    prefillInputName: [.int32],
                    prefillPositionInputName: [.int32],
                    prefillCausalMaskInputName: [.float16]
                ],
                prefillOutputs: [
                    prefillLogitsOutputName: floatingLogits
                ],
                decodeInputs: [
                    decodeTokenInputName: [.int32],
                    decodePositionInputName: [.int32],
                    decodeCausalMaskInputName: [.float16]
                ],
                decodeOutputs: [
                    decodeLogitsOutputName: floatingLogits
                ]
            )
        case .logitsAndLayeredKV(let layerCount, _, _):
            let floatingLogits: [MLMultiArrayDataType] = [.float16, .float32, .double]
            var prefillOutputs: [String: [MLMultiArrayDataType]] = [
                prefillLogitsOutputName: floatingLogits
            ]
            var decodeInputs: [String: [MLMultiArrayDataType]] = [
                decodeTokenInputName: [.int32],
                decodePositionInputName: [.int32],
                decodeCausalMaskInputName: [.float16]
            ]
            var decodeOutputs: [String: [MLMultiArrayDataType]] = [
                decodeLogitsOutputName: floatingLogits
            ]

            for layer in 0..<layerCount {
                prefillOutputs[prefillKeyOutputName(forLayer: layer)] = [.float16]
                prefillOutputs[prefillValueOutputName(forLayer: layer)] = [.float16]
                decodeInputs[decodePastKeyInputName(forLayer: layer)] = [.float16]
                decodeInputs[decodePastValueInputName(forLayer: layer)] = [.float16]
                decodeOutputs[decodeNewKeyOutputName(forLayer: layer)] = [.float16]
                decodeOutputs[decodeNewValueOutputName(forLayer: layer)] = [.float16]
            }

            return (
                prefillInputs: [
                    prefillInputName: [.int32],
                    prefillPositionInputName: [.int32],
                    prefillCausalMaskInputName: [.float16]
                ],
                prefillOutputs: prefillOutputs,
                decodeInputs: decodeInputs,
                decodeOutputs: decodeOutputs
            )
        }
    }
}

private func missingGroup(
    _ label: String,
    _ requiredNames: [String],
    available: Set<String>
) -> String? {
    let missing = requiredNames.filter { !available.contains($0) }
    guard !missing.isEmpty else {
        return nil
    }

    return "\(label): \(missing.joined(separator: ", "))"
}

private func shapeMismatchGroup(
    _ label: String,
    _ expectedShapes: [String: [Int?]],
    actual: [String: [Int]]
) -> String? {
    let mismatches = expectedShapes.keys.sorted().compactMap { name -> String? in
        guard let expectedShape = expectedShapes[name] else {
            return nil
        }
        guard let actualShape = actual[name] else {
            return "\(name) shape unavailable expected \(renderShape(expectedShape))"
        }
        guard shape(actualShape, matches: expectedShape) else {
            return "\(name) shape \(actualShape) expected \(renderShape(expectedShape))"
        }
        return nil
    }

    guard !mismatches.isEmpty else {
        return nil
    }

    return "\(label) \(mismatches.joined(separator: ", "))"
}

private func shape(_ actual: [Int], matches expected: [Int?]) -> Bool {
    guard actual.count == expected.count else {
        return false
    }

    return zip(actual, expected).allSatisfy { actualDimension, expectedDimension in
        guard let expectedDimension else {
            return actualDimension > 0
        }
        return actualDimension == expectedDimension
    }
}

private func renderShape(_ shape: [Int?]) -> String {
    let dimensions = shape.map { dimension in
        dimension.map(String.init) ?? "*"
    }
    return "[\(dimensions.joined(separator: ", "))]"
}

private func dataTypeMismatchGroup(
    _ label: String,
    _ expectedDataTypes: [String: [MLMultiArrayDataType]],
    actual: [String: MLMultiArrayDataType]
) -> String? {
    let mismatches = expectedDataTypes.keys.sorted().compactMap { name -> String? in
        guard let expectedTypes = expectedDataTypes[name] else {
            return nil
        }
        guard let actualType = actual[name] else {
            return "\(name) dtype unavailable expected \(renderDataTypes(expectedTypes))"
        }
        guard expectedTypes.contains(actualType) else {
            return "\(name) dtype \(renderDataType(actualType)) expected \(renderDataTypes(expectedTypes))"
        }
        return nil
    }

    guard !mismatches.isEmpty else {
        return nil
    }

    return "\(label) \(mismatches.joined(separator: ", "))"
}

private func renderDataTypes(_ dataTypes: [MLMultiArrayDataType]) -> String {
    guard dataTypes.count != 1, !dataTypes.isEmpty else {
        return dataTypes.first.map(renderDataType) ?? "unavailable"
    }

    return "[\(dataTypes.map(renderDataType).joined(separator: ", "))]"
}

private func renderDataType(_ dataType: MLMultiArrayDataType) -> String {
    switch dataType {
    case .double:
        return "double"
    case .float32:
        return "float32"
    case .float16:
        return "float16"
    case .int32:
        return "int32"
    default:
        return "unknown(\(dataType.rawValue))"
    }
}

private func multiArrayShapes(
    from descriptions: [String: MLFeatureDescription]
) -> [String: [Int]] {
    descriptions.reduce(into: [:]) { shapes, item in
        guard let shape = item.value.multiArrayConstraint?.shape else {
            return
        }
        shapes[item.key] = shape.map(\.intValue)
    }
}

private func multiArrayDataTypes(
    from descriptions: [String: MLFeatureDescription]
) -> [String: MLMultiArrayDataType] {
    descriptions.reduce(into: [:]) { dataTypes, item in
        guard let dataType = item.value.multiArrayConstraint?.dataType else {
            return
        }
        dataTypes[item.key] = dataType
    }
}
#endif
