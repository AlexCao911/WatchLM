import Foundation

public enum ModelAssetStagingPurpose: String, Codable, Equatable, Sendable {
    case manifest
    case prefill
    case decode
    case tokenizer
}

public struct ModelAssetStagingItem: Codable, Equatable, Sendable {
    public var purposes: [ModelAssetStagingPurpose]
    public var sourceURL: URL
    public var destinationRelativePath: String
    public var byteCount: Int64
    public var expectedSHA256: String?
    public var actualSHA256: String?

    public init(
        purposes: [ModelAssetStagingPurpose],
        sourceURL: URL,
        destinationRelativePath: String,
        byteCount: Int64,
        expectedSHA256: String? = nil,
        actualSHA256: String? = nil
    ) {
        self.purposes = purposes
        self.sourceURL = sourceURL
        self.destinationRelativePath = destinationRelativePath
        self.byteCount = byteCount
        self.expectedSHA256 = expectedSHA256
        self.actualSHA256 = actualSHA256
    }
}

public struct ModelAssetStagingPlan: Codable, Equatable, Sendable {
    public var deviceProfile: DeviceProfile
    public var contextVariant: Int
    public var destinationRootDescription: String
    public var items: [ModelAssetStagingItem]

    public init(
        deviceProfile: DeviceProfile,
        contextVariant: Int,
        destinationRootDescription: String,
        items: [ModelAssetStagingItem]
    ) {
        self.deviceProfile = deviceProfile
        self.contextVariant = contextVariant
        self.destinationRootDescription = destinationRootDescription
        self.items = items
    }

    public var totalByteCount: Int64 {
        items.reduce(0) { $0 + $1.byteCount }
    }
}

public extension ModelAssetStore {
    func stagingPlan(
        for manifest: ModelManifest,
        deviceProfile: DeviceProfile,
        requestedContextTokens: Int? = nil,
        manifestSourceURL: URL? = nil,
        destinationRootDescription: String = "Application Support/WatchLM"
    ) throws -> ModelAssetStagingPlan {
        let artifact = try selectedArtifact(
            for: manifest,
            deviceProfile: deviceProfile,
            requestedContextTokens: requestedContextTokens
        )

        var items: [ModelAssetStagingItem] = [
            try stagingItem(
                purposes: [.manifest],
                sourceURL: manifestSourceURL ?? manifestURL,
                destinationRelativePath: manifestFileName,
                expectedSHA256: nil
            )
        ]

        try appendArtifactStagingItem(
            to: &items,
            purpose: .prefill,
            relativePath: artifact.prefillPath,
            expectedSHA256: artifact.prefillSHA256 ?? artifact.sha256
        )
        try appendArtifactStagingItem(
            to: &items,
            purpose: .decode,
            relativePath: artifact.decodePath,
            expectedSHA256: artifact.decodeSHA256 ?? artifact.sha256
        )

        if let tokenizerPath = artifact.tokenizerPath {
            try appendArtifactStagingItem(
                to: &items,
                purpose: .tokenizer,
                relativePath: tokenizerPath,
                expectedSHA256: artifact.tokenizerSHA256
            )
        }

        return ModelAssetStagingPlan(
            deviceProfile: deviceProfile,
            contextVariant: artifact.contextVariant,
            destinationRootDescription: destinationRootDescription,
            items: items
        )
    }

    private func appendArtifactStagingItem(
        to items: inout [ModelAssetStagingItem],
        purpose: ModelAssetStagingPurpose,
        relativePath: String,
        expectedSHA256: String?
    ) throws {
        if let existingIndex = items.firstIndex(where: { $0.destinationRelativePath == relativePath }) {
            if !items[existingIndex].purposes.contains(purpose) {
                items[existingIndex].purposes.append(purpose)
            }
            if items[existingIndex].expectedSHA256 == nil {
                items[existingIndex].expectedSHA256 = expectedSHA256
            }
            return
        }

        items.append(try stagingItem(
            purposes: [purpose],
            relativePath: relativePath,
            expectedSHA256: expectedSHA256
        ))
    }

    private func stagingItem(
        purposes: [ModelAssetStagingPurpose],
        relativePath: String,
        expectedSHA256: String?
    ) throws -> ModelAssetStagingItem {
        try validateStagingRelativePath(relativePath)
        let sourceURL = url(for: relativePath).resolvingSymlinksInPath()
        return try stagingItem(
            purposes: purposes,
            sourceURL: sourceURL,
            destinationRelativePath: relativePath,
            expectedSHA256: expectedSHA256
        )
    }

    private func stagingItem(
        purposes: [ModelAssetStagingPurpose],
        sourceURL: URL,
        destinationRelativePath: String,
        expectedSHA256: String?
    ) throws -> ModelAssetStagingItem {
        try validateStagingRelativePath(destinationRelativePath)
        let sourceURL = sourceURL.resolvingSymlinksInPath()
        return try ModelAssetStagingItem(
            purposes: purposes,
            sourceURL: sourceURL,
            destinationRelativePath: destinationRelativePath,
            byteCount: Self.byteCount(for: sourceURL),
            expectedSHA256: expectedSHA256,
            actualSHA256: ArtifactDigest.sha256Hex(for: sourceURL)
        )
    }

    private func validateStagingRelativePath(_ relativePath: String) throws {
        guard !relativePath.isEmpty,
              !relativePath.hasPrefix("/"),
              !relativePath.split(separator: "/").contains("..")
        else {
            throw InferenceRuntimeError.invalidInput(message: "Invalid staging relative path \(relativePath).")
        }
    }

    private static func byteCount(for url: URL) throws -> Int64 {
        let values = try url.resourceValues(forKeys: [.isDirectoryKey, .fileSizeKey])
        if values.isDirectory == true {
            return try directoryByteCount(for: url)
        }

        return Int64(values.fileSize ?? 0)
    }

    private static func directoryByteCount(for directoryURL: URL) throws -> Int64 {
        guard let enumerator = FileManager.default.enumerator(
            at: directoryURL,
            includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else {
            throw CocoaError(.fileReadUnknown)
        }

        var total: Int64 = 0
        for case let fileURL as URL in enumerator {
            let values = try fileURL.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
            if values.isRegularFile == true {
                total += Int64(values.fileSize ?? 0)
            }
        }
        return total
    }
}
