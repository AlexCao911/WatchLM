public struct ModelManifest: Codable, Equatable, Sendable {
    public var schemaVersion: Int
    public var model: ModelInfo
    public var runtime: RuntimeInfo
    public var architecture: ArchitectureInfo
    public var deviceProfiles: [String: DeviceProfileConfiguration]
    public var contextVariants: [Int]
    public var asset: AssetInfo
    public var quantization: QuantizationInfo
    public var fallbackPolicy: FallbackPolicy

    public var validationErrors: [String] {
        var errors: [String] = []

        if model.id != ModelManifestContract.expectedModelId {
            errors.append("model.id must be \(ModelManifestContract.expectedModelId)")
        }

        if runtime.type != ModelManifestContract.expectedRuntime {
            errors.append("runtime.type must be \(ModelManifestContract.expectedRuntime)")
        }

        if !runtime.entrypoints.contains("prefill") || !runtime.entrypoints.contains("decode") {
            errors.append("runtime.entrypoints must include prefill and decode")
        }

        if !ModelManifestContract.supportedKVCacheModes.contains(runtime.kvCacheMode) {
            errors.append("runtime.kvCacheMode must be stateful-preferred, slot-ring, or contiguous-sliding")
        }

        validateRuntimeGraphSchema(into: &errors)

        if architecture.layers != ModelManifestContract.layers {
            errors.append("architecture.layers must be \(ModelManifestContract.layers)")
        }

        if architecture.hiddenSize != ModelManifestContract.hiddenSize {
            errors.append("architecture.hiddenSize must be \(ModelManifestContract.hiddenSize)")
        }

        if architecture.queryHeads != ModelManifestContract.queryHeads {
            errors.append("architecture.queryHeads must be \(ModelManifestContract.queryHeads)")
        }

        if architecture.kvHeads != ModelManifestContract.kvHeads {
            errors.append("architecture.kvHeads must be \(ModelManifestContract.kvHeads)")
        }

        if !architecture.tokenizer.preserved {
            errors.append("tokenizer must be preserved")
        }

        if !architecture.tokenizer.vocabularyPreserved {
            errors.append("vocabulary must be preserved")
        }

        for variant in contextVariants where !ModelManifestContract.supportedContextVariants.contains(variant) {
            errors.append("unsupported context variant \(variant)")
        }

        if asset.storage == "app-bundle" {
            errors.append("asset.storage must not be app-bundle")
        }

        validateStatefulSharedArtifacts(into: &errors)

        if let variants = asset.variants {
            for profile in DeviceProfile.allCases {
                guard let defaultContextVariant = deviceProfiles[profile.rawValue]?.defaultContextVariant else {
                    continue
                }
                guard let variant = variants[String(defaultContextVariant)] else {
                    errors.append("asset.variants.\(defaultContextVariant) must be present for \(profile.rawValue)")
                    continue
                }

                if variant.deviceProfile != profile.rawValue {
                    errors.append("asset.variants.\(defaultContextVariant).deviceProfile must be \(profile.rawValue)")
                }
            }
        }

        if quantization.strategy != "mixed-precision-fidelity-first" {
            errors.append("quantization.strategy must be mixed-precision-fidelity-first")
        }

        if quantization.kvCache != "fp16" && quantization.kvCache != "int8" {
            errors.append("quantization.kvCache must be fp16 or int8")
        }

        if quantization.structuralReduction {
            errors.append("structuralReduction must be false")
        }

        return errors
    }

    private func validateRuntimeGraphSchema(into errors: inout [String]) {
        let graphSchema = runtime.graphSchema

        if !ModelManifestContract.supportedGraphInterfaces.contains(graphSchema.interface) {
            errors.append("runtime.graphSchema.interface must be logits-layered-kv, stateful-kv, or stateful-step-kv")
        }

        if graphSchema.layerCount != ModelManifestContract.layers {
            errors.append("runtime.graphSchema.layerCount must be \(ModelManifestContract.layers)")
        }

        if graphSchema.kvHeads != ModelManifestContract.kvHeads {
            errors.append("runtime.graphSchema.kvHeads must be \(ModelManifestContract.kvHeads)")
        }

        if graphSchema.headDimension != ModelManifestContract.headDimension {
            errors.append("runtime.graphSchema.headDimension must be \(ModelManifestContract.headDimension)")
        }

        if graphSchema.prefill.inputIDs != ModelManifestContract.prefillInputIDs {
            errors.append("runtime.graphSchema.prefill.inputIDs must be \(ModelManifestContract.prefillInputIDs)")
        }

        if graphSchema.prefill.positionIDs != ModelManifestContract.prefillPositionIDs {
            errors.append("runtime.graphSchema.prefill.positionIDs must be \(ModelManifestContract.prefillPositionIDs)")
        }

        if graphSchema.prefill.causalMask != ModelManifestContract.causalMask {
            errors.append("runtime.graphSchema.prefill.causalMask must be \(ModelManifestContract.causalMask)")
        }

        if graphSchema.prefill.logits != ModelManifestContract.logits {
            errors.append("runtime.graphSchema.prefill.logits must be \(ModelManifestContract.logits)")
        }

        if graphSchema.prefill.keyPrefix != ModelManifestContract.prefillKeyPrefix {
            errors.append("runtime.graphSchema.prefill.keyPrefix must be \(ModelManifestContract.prefillKeyPrefix)")
        }

        if graphSchema.prefill.valuePrefix != ModelManifestContract.prefillValuePrefix {
            errors.append("runtime.graphSchema.prefill.valuePrefix must be \(ModelManifestContract.prefillValuePrefix)")
        }

        let expectedDecodeTokenID = ModelManifestContract.expectedDecodeTokenID(for: graphSchema.interface)
        if graphSchema.decode.tokenID != expectedDecodeTokenID {
            errors.append("runtime.graphSchema.decode.tokenID must be \(expectedDecodeTokenID)")
        }

        let expectedDecodePositionID = ModelManifestContract.expectedDecodePositionID(for: graphSchema.interface)
        if graphSchema.decode.positionID != expectedDecodePositionID {
            errors.append("runtime.graphSchema.decode.positionID must be \(expectedDecodePositionID)")
        }

        if graphSchema.decode.causalMask != ModelManifestContract.causalMask {
            errors.append("runtime.graphSchema.decode.causalMask must be \(ModelManifestContract.causalMask)")
        }

        if graphSchema.decode.logits != ModelManifestContract.logits {
            errors.append("runtime.graphSchema.decode.logits must be \(ModelManifestContract.logits)")
        }

        if graphSchema.decode.pastKeyPrefix != ModelManifestContract.decodePastKeyPrefix {
            errors.append("runtime.graphSchema.decode.pastKeyPrefix must be \(ModelManifestContract.decodePastKeyPrefix)")
        }

        if graphSchema.decode.pastValuePrefix != ModelManifestContract.decodePastValuePrefix {
            errors.append("runtime.graphSchema.decode.pastValuePrefix must be \(ModelManifestContract.decodePastValuePrefix)")
        }

        if graphSchema.decode.newKeyPrefix != ModelManifestContract.decodeNewKeyPrefix {
            errors.append("runtime.graphSchema.decode.newKeyPrefix must be \(ModelManifestContract.decodeNewKeyPrefix)")
        }

        if graphSchema.decode.newValuePrefix != ModelManifestContract.decodeNewValuePrefix {
            errors.append("runtime.graphSchema.decode.newValuePrefix must be \(ModelManifestContract.decodeNewValuePrefix)")
        }
    }

    private func validateStatefulSharedArtifacts(into errors: inout [String]) {
        guard ModelManifestContract.statefulGraphInterfaces.contains(runtime.graphSchema.interface) else {
            return
        }

        if asset.prefillPath != asset.decodePath {
            errors.append("stateful Core ML graphs must use the same artifact path for prefill and decode")
        }

        for (contextVariant, variant) in asset.variants ?? [:] where variant.prefillPath != variant.decodePath {
            errors.append("asset.variants.\(contextVariant) must use the same artifact path for prefill and decode for stateful Core ML graphs")
        }
    }

    public func modelArtifact(
        for deviceProfile: DeviceProfile,
        requestedContextTokens: Int?
    ) throws -> SelectedModelArtifact {
        guard let profile = deviceProfiles[deviceProfile.rawValue] else {
            throw InferenceRuntimeError.invalidInput(message: "Unsupported device profile \(deviceProfile.rawValue).")
        }

        let requested = requestedContextTokens ?? profile.defaultContextVariant
        let sortedContextVariants = contextVariants.sorted()
        guard let fallbackContext = sortedContextVariants.first else {
            throw InferenceRuntimeError.invalidInput(message: "Manifest has no context variants.")
        }
        let selectedContext = sortedContextVariants
            .filter { $0 <= requested }
            .last ?? fallbackContext

        if let variant = asset.variants?[String(selectedContext)] {
            return SelectedModelArtifact(
                contextVariant: selectedContext,
                deviceProfile: variant.deviceProfile,
                prefillPath: variant.prefillPath,
                decodePath: variant.decodePath,
                tokenizerPath: variant.tokenizerPath ?? asset.tokenizerPath,
                sha256: variant.sha256,
                prefillSHA256: variant.prefillSHA256,
                decodeSHA256: variant.decodeSHA256,
                tokenizerSHA256: variant.tokenizerSHA256 ?? asset.tokenizerSHA256
            )
        }

        return SelectedModelArtifact(
            contextVariant: selectedContext,
            deviceProfile: deviceProfile.rawValue,
            prefillPath: asset.prefillPath,
            decodePath: asset.decodePath,
            tokenizerPath: asset.tokenizerPath,
            sha256: asset.sha256,
            prefillSHA256: asset.prefillSHA256,
            decodeSHA256: asset.decodeSHA256,
            tokenizerSHA256: asset.tokenizerSHA256
        )
    }
}

public struct ModelInfo: Codable, Equatable, Sendable {
    public var id: String
    public var revision: String
    public var parameterCount: Int
}

public struct RuntimeInfo: Codable, Equatable, Sendable {
    public var type: String
    public var entrypoints: [String]
    public var kvCacheMode: String
    public var graphSchema: RuntimeGraphSchema

    public func kvCacheRouteDecision(
        capabilities: CoreMLRuntimeCapabilities
    ) -> CoreMLKVCacheRouteDecision {
        CoreMLKVCacheRoutePlanner.selectRoute(
            kvCacheMode: kvCacheMode,
            graphInterface: graphSchema.interface,
            capabilities: capabilities
        )
    }

    public var kvCacheUpdateStrategy: KVCacheUpdateStrategy {
        switch kvCacheMode {
        case "contiguous-sliding":
            return .contiguousSliding
        case "stateful-preferred", "slot-ring":
            return .slotRing
        default:
            return .slotRing
        }
    }
}

public struct RuntimeGraphSchema: Codable, Equatable, Sendable {
    public var interface: String
    public var layerCount: Int
    public var kvHeads: Int
    public var headDimension: Int
    public var prefill: PrefillGraphSchema
    public var decode: DecodeGraphSchema
}

public struct PrefillGraphSchema: Codable, Equatable, Sendable {
    public var inputIDs: String
    public var positionIDs: String
    public var causalMask: String
    public var logits: String
    public var keyPrefix: String
    public var valuePrefix: String
}

public struct DecodeGraphSchema: Codable, Equatable, Sendable {
    public var tokenID: String
    public var positionID: String
    public var causalMask: String
    public var logits: String
    public var pastKeyPrefix: String
    public var pastValuePrefix: String
    public var newKeyPrefix: String
    public var newValuePrefix: String
}

public struct ArchitectureInfo: Codable, Equatable, Sendable {
    public var type: String
    public var layers: Int
    public var hiddenSize: Int
    public var queryHeads: Int
    public var kvHeads: Int
    public var maxContextTokens: Int
    public var tokenizer: TokenizerInfo
}

public struct TokenizerInfo: Codable, Equatable, Sendable {
    public var source: String
    public var preserved: Bool
    public var vocabularyPreserved: Bool
    public var chatTemplate: String
}

public struct DeviceProfileConfiguration: Codable, Equatable, Sendable {
    public var sip: String
    public var neuralEngineCores: Int
    public var defaultContextVariant: Int
    public var maxNewTokens: Int
}

public struct AssetInfo: Codable, Equatable, Sendable {
    public var storage: String
    public var prefillPath: String
    public var decodePath: String
    public var tokenizerPath: String?
    public var sha256: String
    public var prefillSHA256: String?
    public var decodeSHA256: String?
    public var tokenizerSHA256: String?
    public var variants: [String: ModelArtifactVariant]?
}

public struct ModelArtifactVariant: Codable, Equatable, Sendable {
    public var deviceProfile: String
    public var prefillPath: String
    public var decodePath: String
    public var tokenizerPath: String?
    public var sha256: String
    public var prefillSHA256: String?
    public var decodeSHA256: String?
    public var tokenizerSHA256: String?
}

public struct SelectedModelArtifact: Codable, Equatable, Sendable {
    public var contextVariant: Int
    public var deviceProfile: String
    public var prefillPath: String
    public var decodePath: String
    public var tokenizerPath: String?
    public var sha256: String
    public var prefillSHA256: String?
    public var decodeSHA256: String?
    public var tokenizerSHA256: String?

    public init(
        contextVariant: Int,
        deviceProfile: String,
        prefillPath: String,
        decodePath: String,
        tokenizerPath: String?,
        sha256: String,
        prefillSHA256: String?,
        decodeSHA256: String?,
        tokenizerSHA256: String?
    ) {
        self.contextVariant = contextVariant
        self.deviceProfile = deviceProfile
        self.prefillPath = prefillPath
        self.decodePath = decodePath
        self.tokenizerPath = tokenizerPath
        self.sha256 = sha256
        self.prefillSHA256 = prefillSHA256
        self.decodeSHA256 = decodeSHA256
        self.tokenizerSHA256 = tokenizerSHA256
    }
}

public struct QuantizationInfo: Codable, Equatable, Sendable {
    public var strategy: String
    public var weights: QuantizationWeights
    public var kvCache: String
    public var structuralReduction: Bool
}

public struct QuantizationWeights: Codable, Equatable, Sendable {
    public var embedding: String
    public var lmHead: String
    public var norms: String
    public var attentionQKO: String
    public var attentionV: String
    public var ffn: String
}

public struct FallbackPolicy: Codable, Equatable, Sendable {
    public var requiresBenchmarkEvidence: Bool
    public var order: [String]
}

enum ModelManifestContract {
    static let expectedModelId = "openbmb/MiniCPM5-1B"
    static let expectedRuntime = "coreml-mlprogram"
    static let explicitKVGraphInterface = "logits-layered-kv"
    static let statefulKVGraphInterface = "stateful-kv"
    static let statefulStepKVGraphInterface = "stateful-step-kv"
    static let supportedGraphInterfaces = [
        explicitKVGraphInterface,
        statefulKVGraphInterface,
        statefulStepKVGraphInterface
    ]
    static let statefulGraphInterfaces = [
        statefulKVGraphInterface,
        statefulStepKVGraphInterface
    ]
    static let supportedKVCacheModes = ["stateful-preferred", "slot-ring", "contiguous-sliding"]
    static let layers = 24
    static let hiddenSize = 1536
    static let queryHeads = 16
    static let kvHeads = 2
    static let headDimension = 128
    static let supportedContextVariants = [256, 512, 1024]
    static let prefillInputIDs = "input_ids"
    static let prefillPositionIDs = "position_ids"
    static let causalMask = "causal_mask"
    static let logits = "logits"
    static let prefillKeyPrefix = "present_key_"
    static let prefillValuePrefix = "present_value_"
    static let decodeTokenID = "token_id"
    static let decodePositionID = "position_id"
    static let decodePastKeyPrefix = "past_key_"
    static let decodePastValuePrefix = "past_value_"
    static let decodeNewKeyPrefix = "new_key_"
    static let decodeNewValuePrefix = "new_value_"

    static func expectedDecodeTokenID(for graphInterface: String) -> String {
        graphInterface == statefulStepKVGraphInterface ? prefillInputIDs : decodeTokenID
    }

    static func expectedDecodePositionID(for graphInterface: String) -> String {
        graphInterface == statefulStepKVGraphInterface ? prefillPositionIDs : decodePositionID
    }
}
