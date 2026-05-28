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
    public var sha256: String
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
