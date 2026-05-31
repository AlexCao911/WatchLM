import CryptoKit
import Foundation

public enum ArtifactDigest {
    public static func sha256Hex(for url: URL) throws -> String {
        let resolvedURL = url.resolvingSymlinksInPath()
        if try isDirectory(resolvedURL) {
            return try directorySHA256Hex(for: resolvedURL)
        }

        return try fileSHA256Hex(for: resolvedURL)
    }

    private static func fileSHA256Hex(for fileURL: URL) throws -> String {
        let data = try Data(contentsOf: fileURL)
        let digest = SHA256.hash(data: data)
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    private static func directorySHA256Hex(for directoryURL: URL) throws -> String {
        guard let enumerator = FileManager.default.enumerator(
            at: directoryURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            throw CocoaError(.fileReadUnknown)
        }

        var fileURLs: [URL] = []
        for case let fileURL as URL in enumerator {
            let values = try fileURL.resourceValues(forKeys: [.isRegularFileKey])
            if values.isRegularFile == true {
                fileURLs.append(fileURL)
            }
        }

        var hasher = SHA256()
        for fileURL in fileURLs.sorted(by: { relativePath($0, from: directoryURL) < relativePath($1, from: directoryURL) }) {
            let path = relativePath(fileURL, from: directoryURL)
            hasher.update(data: Data(path.utf8))
            hasher.update(data: Data([0]))
            hasher.update(data: try Data(contentsOf: fileURL))
            hasher.update(data: Data([0]))
        }

        let digest = hasher.finalize()
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    private static func isDirectory(_ url: URL) throws -> Bool {
        let values = try url.resourceValues(forKeys: [.isDirectoryKey])
        return values.isDirectory == true
    }

    private static func relativePath(_ fileURL: URL, from directoryURL: URL) -> String {
        let directoryPath = directoryURL.standardizedFileURL.path
        let filePath = fileURL.standardizedFileURL.path
        guard filePath.hasPrefix(directoryPath) else {
            return fileURL.lastPathComponent
        }

        var relative = String(filePath.dropFirst(directoryPath.count))
        if relative.hasPrefix("/") {
            relative.removeFirst()
        }
        return relative
    }
}
