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
