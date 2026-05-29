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

        if quantization.kvCache != "int8" {
            errors.append("quantization.kvCache must be int8")
        }

        if quantization.structuralReduction {
            errors.append("structuralReduction must be false")
        }

        return errors
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
    static let layers = 24
    static let hiddenSize = 1536
    static let queryHeads = 16
    static let kvHeads = 2
    static let supportedContextVariants = [256, 512, 1024]
}
