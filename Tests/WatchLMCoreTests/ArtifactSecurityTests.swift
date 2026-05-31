import Foundation
import Testing
@testable import WatchLMCore

@Test func artifactDigestComputesSHA256Hex() throws {
    let directory = try temporaryDirectory()
    let fileURL = directory.appending(path: "tokenizer.json")
    try Data("watchlm".utf8).write(to: fileURL)

    let digest = try ArtifactDigest.sha256Hex(for: fileURL)

    #expect(digest == "3137b156710585c5773fa1b9f83a0ad78e550e82f09e32439db0d48b71490345")
}

@Test func artifactDigestComputesStableDirectorySHA256Hex() throws {
    let firstPackage = try makePackageDirectory(files: [
        "Manifest.json": "manifest",
        "weights/weight.bin": "weights"
    ])
    let secondPackage = try makePackageDirectory(files: [
        "weights/weight.bin": "weights",
        "Manifest.json": "manifest"
    ])

    let firstDigest = try ArtifactDigest.sha256Hex(for: firstPackage)
    let secondDigest = try ArtifactDigest.sha256Hex(for: secondPackage)

    #expect(firstDigest == secondDigest)

    try Data("changed".utf8).write(to: secondPackage.appending(path: "weights/weight.bin"))
    #expect(try ArtifactDigest.sha256Hex(for: secondPackage) != firstDigest)
}

@Test func artifactDigestFollowsSymlinkedArtifacts() throws {
    let targetPackage = try makePackageDirectory(files: [
        "Manifest.json": "manifest",
        "weights/weight.bin": "weights"
    ])
    let linkedPackage = try temporaryDirectory()
        .appending(path: "linked.mlpackage", directoryHint: .isDirectory)
    try FileManager.default.createSymbolicLink(at: linkedPackage, withDestinationURL: targetPackage)

    let targetTokenizer = try temporaryDirectory().appending(path: "tokenizer.json")
    let linkedTokenizer = try temporaryDirectory().appending(path: "tokenizer.json")
    try Data("tokenizer".utf8).write(to: targetTokenizer)
    try FileManager.default.createSymbolicLink(at: linkedTokenizer, withDestinationURL: targetTokenizer)

    #expect(try ArtifactDigest.sha256Hex(for: linkedPackage) == ArtifactDigest.sha256Hex(for: targetPackage))
    #expect(try ArtifactDigest.sha256Hex(for: linkedTokenizer) == ArtifactDigest.sha256Hex(for: targetTokenizer))
}

@Test func modelArtifactVerifierReportsMissingAndMismatchedFiles() throws {
    let directory = try temporaryDirectory()
    let prefillURL = directory.appending(path: "prefill-256.mlpackage")
    let tokenizerURL = directory.appending(path: "tokenizer.json")
    try Data("prefill".utf8).write(to: prefillURL)
    try Data("tokenizer".utf8).write(to: tokenizerURL)

    let artifact = SelectedModelArtifact(
        contextVariant: 256,
        deviceProfile: "watch-se-2",
        prefillPath: "prefill-256.mlpackage",
        decodePath: "decode-256.mlpackage",
        tokenizerPath: "tokenizer.json",
        sha256: "aggregate",
        prefillSHA256: try ArtifactDigest.sha256Hex(for: prefillURL),
        decodeSHA256: "1111111111111111111111111111111111111111111111111111111111111111",
        tokenizerSHA256: "2222222222222222222222222222222222222222222222222222222222222222"
    )

    let report = ModelArtifactVerifier().verify(
        artifact: artifact,
        relativeTo: directory
    )

    #expect(!report.isReady)
    #expect(report.finding(for: .prefill)?.status == .verified)
    #expect(report.finding(for: .decode)?.status == .missing)
    #expect(report.finding(for: .tokenizer)?.status == .hashMismatch)
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appending(path: "watchlm-\(UUID().uuidString)", directoryHint: .isDirectory)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

private func makePackageDirectory(files: [String: String]) throws -> URL {
    let packageURL = try temporaryDirectory().appending(path: "model.mlpackage", directoryHint: .isDirectory)
    try FileManager.default.createDirectory(at: packageURL, withIntermediateDirectories: true)
    for (path, contents) in files {
        let fileURL = packageURL.appending(path: path)
        try FileManager.default.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try Data(contents.utf8).write(to: fileURL)
    }
    return packageURL
}
