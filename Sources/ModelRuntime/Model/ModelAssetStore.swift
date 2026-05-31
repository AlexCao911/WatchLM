import Foundation

public struct ModelAssetStore: Sendable {
    public var rootURL: URL
    public var manifestFileName: String

    public init(
        rootURL: URL,
        manifestFileName: String = "model-manifest.json"
    ) {
        self.rootURL = rootURL
        self.manifestFileName = manifestFileName
    }

    public static func defaultRootURL(
        folderName: String = "WatchLM",
        fileManager: FileManager = .default
    ) throws -> URL {
        let applicationSupportURL = try fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let rootURL = applicationSupportURL.appending(path: folderName, directoryHint: .isDirectory)
        try fileManager.createDirectory(at: rootURL, withIntermediateDirectories: true)
        return rootURL
    }

    public var manifestURL: URL {
        rootURL.appending(path: manifestFileName)
    }

    public func url(for relativePath: String) -> URL {
        rootURL.appending(path: relativePath)
    }

    public func loadManifest() throws -> ModelManifest {
        let data = try Data(contentsOf: manifestURL)
        return try JSONDecoder().decode(ModelManifest.self, from: data)
    }

    public func saveManifest(_ manifest: ModelManifest) throws {
        try FileManager.default.createDirectory(at: rootURL, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(manifest).write(to: manifestURL)
    }

    public func selectedArtifact(
        for manifest: ModelManifest,
        deviceProfile: DeviceProfile,
        requestedContextTokens: Int? = nil
    ) throws -> SelectedModelArtifact {
        try manifest.modelArtifact(
            for: deviceProfile,
            requestedContextTokens: requestedContextTokens
        )
    }

    public func verificationReport(
        for manifest: ModelManifest,
        deviceProfile: DeviceProfile,
        requestedContextTokens: Int? = nil,
        verifier: ModelArtifactVerifier = ModelArtifactVerifier()
    ) throws -> ModelArtifactVerificationReport {
        let artifact = try selectedArtifact(
            for: manifest,
            deviceProfile: deviceProfile,
            requestedContextTokens: requestedContextTokens
        )
        return verifier.verify(artifact: artifact, relativeTo: rootURL)
    }

    public func assetState(
        for manifest: ModelManifest,
        deviceProfile: DeviceProfile,
        requestedContextTokens: Int? = nil,
        runtimeCapabilities: CoreMLRuntimeCapabilities = .current,
        verifier: ModelArtifactVerifier = ModelArtifactVerifier()
    ) -> ModelAssetState {
        let validationErrors = manifest.validationErrors
        guard validationErrors.isEmpty else {
            return .incompatibleManifest(errors: validationErrors)
        }

        let routeDecision = manifest.runtime.kvCacheRouteDecision(
            capabilities: runtimeCapabilities
        )
        guard routeDecision.selectedRoute != .unsupportedStatefulKV else {
            return .unavailableRuntime(reason: routeDecision.reason)
        }

        let report: ModelArtifactVerificationReport
        do {
            report = try verificationReport(
                for: manifest,
                deviceProfile: deviceProfile,
                requestedContextTokens: requestedContextTokens,
                verifier: verifier
            )
        } catch {
            return .incompatibleManifest(errors: [String(describing: error)])
        }

        if report.isReady {
            return .installed(manifest: manifest)
        }

        if let mismatch = report.findings.first(where: { $0.status == .hashMismatch }) {
            return .invalidHash(
                expected: mismatch.expectedSHA256 ?? "",
                actual: mismatch.actualSHA256 ?? ""
            )
        }

        if report.findings.contains(where: { $0.status == .missing }) {
            return .missing
        }

        return .unavailableRuntime(reason: "Model artifact hashes are unavailable.")
    }
}
