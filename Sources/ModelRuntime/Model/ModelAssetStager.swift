import Foundation

public struct ModelAssetStagingCopiedItem: Codable, Equatable, Sendable {
    public var destinationRelativePath: String
    public var byteCount: Int64
    public var actualSHA256: String

    public init(
        destinationRelativePath: String,
        byteCount: Int64,
        actualSHA256: String
    ) {
        self.destinationRelativePath = destinationRelativePath
        self.byteCount = byteCount
        self.actualSHA256 = actualSHA256
    }
}

public struct ModelAssetStagingResult: Codable, Equatable, Sendable {
    public var destinationRootURL: URL
    public var copiedItems: [ModelAssetStagingCopiedItem]

    public init(
        destinationRootURL: URL,
        copiedItems: [ModelAssetStagingCopiedItem]
    ) {
        self.destinationRootURL = destinationRootURL
        self.copiedItems = copiedItems
    }

    public var itemCount: Int {
        copiedItems.count
    }

    public var totalByteCount: Int64 {
        copiedItems.reduce(0) { $0 + $1.byteCount }
    }
}

public struct ModelAssetStager {
    private let fileManager: FileManager

    public init(fileManager: FileManager = .default) {
        self.fileManager = fileManager
    }

    public func stage(
        plan: ModelAssetStagingPlan,
        to destinationRootURL: URL
    ) throws -> ModelAssetStagingResult {
        try fileManager.createDirectory(
            at: destinationRootURL,
            withIntermediateDirectories: true
        )

        let copiedItems = try plan.items.map { item in
            try copy(item: item, to: destinationRootURL)
        }
        return ModelAssetStagingResult(
            destinationRootURL: destinationRootURL,
            copiedItems: copiedItems
        )
    }

    private func copy(
        item: ModelAssetStagingItem,
        to destinationRootURL: URL
    ) throws -> ModelAssetStagingCopiedItem {
        try validateRelativePath(item.destinationRelativePath)
        let sourceURL = item.sourceURL.resolvingSymlinksInPath()
        let destinationURL = destinationRootURL.appending(path: item.destinationRelativePath)
        let destinationParentURL = destinationURL.deletingLastPathComponent()
        try fileManager.createDirectory(
            at: destinationParentURL,
            withIntermediateDirectories: true
        )

        if fileManager.fileExists(atPath: destinationURL.path) {
            try fileManager.removeItem(at: destinationURL)
        }
        try fileManager.copyItem(at: sourceURL, to: destinationURL)

        let actualSHA256 = try ArtifactDigest.sha256Hex(for: destinationURL)
        if let expectedSHA256 = item.expectedSHA256, expectedSHA256 != actualSHA256 {
            throw InferenceRuntimeError.invalidInput(
                message: "Staged artifact hash mismatch at \(item.destinationRelativePath): expected \(expectedSHA256), got \(actualSHA256)."
            )
        }

        return try ModelAssetStagingCopiedItem(
            destinationRelativePath: item.destinationRelativePath,
            byteCount: Self.byteCount(for: destinationURL),
            actualSHA256: actualSHA256
        )
    }

    private func validateRelativePath(_ relativePath: String) throws {
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
