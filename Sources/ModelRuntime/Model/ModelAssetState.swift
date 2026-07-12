public enum ModelAssetState: Codable, Equatable, Sendable {
    case missing
    case installing(progress: Double)
    case installed(manifest: ModelManifest)
    case invalidHash(expected: String, actual: String)
    case incompatibleManifest(errors: [String])
    case unavailableRuntime(reason: String)
}
