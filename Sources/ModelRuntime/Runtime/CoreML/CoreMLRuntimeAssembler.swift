#if canImport(CoreML)
import Foundation

public enum CoreMLRuntimeAssemblyError: Error, Equatable, Sendable {
    case invalidManifest([String])
    case artifactVerificationFailed(ModelArtifactVerificationReport)
    case missingTokenizerPath
}

public struct CoreMLRuntimeAssembly: Sendable {
    public var artifact: SelectedModelArtifact
    public var verificationReport: ModelArtifactVerificationReport
    public var bundle: CoreMLPrefillDecodeBundle
    public var tokenizer: MiniCPMBytePairTokenizer
    public var tokenizerURL: URL

    public var prefillModelURL: URL {
        bundle.prefillModelURL
    }

    public var decodeModelURL: URL {
        bundle.decodeModelURL
    }

    public func makeRuntime() -> CoreMLPrefillDecodeRuntime {
        CoreMLPrefillDecodeRuntime(bundle: bundle, tokenizer: tokenizer)
    }
}

public struct CoreMLRuntimeAssembler: Sendable {
    private let verifier: ModelArtifactVerifier

    public init(verifier: ModelArtifactVerifier = ModelArtifactVerifier()) {
        self.verifier = verifier
    }

    public func assemble(
        manifest: ModelManifest,
        deviceProfile: DeviceProfile,
        requestedContextTokens: Int? = nil,
        assetBaseURL: URL,
        logitsProcessor: LogitsProcessor = LogitsProcessor(),
        samplingStrategy: TokenSamplingStrategy = .greedy,
        verifyArtifacts: Bool = true
    ) throws -> CoreMLRuntimeAssembly {
        let validationErrors = manifest.validationErrors
        guard validationErrors.isEmpty else {
            throw CoreMLRuntimeAssemblyError.invalidManifest(validationErrors)
        }

        let artifact = try manifest.modelArtifact(
            for: deviceProfile,
            requestedContextTokens: requestedContextTokens
        )
        let verificationReport = verifier.verify(
            artifact: artifact,
            relativeTo: assetBaseURL
        )
        if verifyArtifacts, !verificationReport.isReady {
            throw CoreMLRuntimeAssemblyError.artifactVerificationFailed(verificationReport)
        }

        guard let tokenizerPath = artifact.tokenizerPath, !tokenizerPath.isEmpty else {
            throw CoreMLRuntimeAssemblyError.missingTokenizerPath
        }

        let prefillModelURL = assetBaseURL.appending(path: artifact.prefillPath, directoryHint: .isDirectory)
        let decodeModelURL = assetBaseURL.appending(path: artifact.decodePath, directoryHint: .isDirectory)
        let tokenizerURL = assetBaseURL.appending(path: tokenizerPath)
        let tokenizer = try MiniCPMBytePairTokenizer(
            tokenizerJSONURL: tokenizerURL,
            addBosToken: true
        )
        let bundle = try CoreMLPrefillDecodeBundle(
            prefillModelURL: prefillModelURL,
            decodeModelURL: decodeModelURL,
            maxPromptTokens: artifact.contextVariant,
            graphSchema: manifest.runtime.graphSchema,
            logitsProcessor: logitsProcessor,
            samplingStrategy: samplingStrategy,
            kvCacheUpdateStrategy: manifest.runtime.kvCacheUpdateStrategy
        )

        return CoreMLRuntimeAssembly(
            artifact: artifact,
            verificationReport: verificationReport,
            bundle: bundle,
            tokenizer: tokenizer,
            tokenizerURL: tokenizerURL
        )
    }
}
#endif
