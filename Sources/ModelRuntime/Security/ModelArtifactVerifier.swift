import Foundation

public enum ModelArtifactComponent: String, Codable, Equatable, Sendable {
    case prefill
    case decode
    case tokenizer
}

public enum ModelArtifactFindingStatus: String, Codable, Equatable, Sendable {
    case verified
    case missing
    case hashMismatch
    case hashUnavailable
}

public struct ModelArtifactFinding: Codable, Equatable, Sendable {
    public var component: ModelArtifactComponent
    public var status: ModelArtifactFindingStatus
    public var path: String
    public var expectedSHA256: String?
    public var actualSHA256: String?

    public init(
        component: ModelArtifactComponent,
        status: ModelArtifactFindingStatus,
        path: String,
        expectedSHA256: String?,
        actualSHA256: String?
    ) {
        self.component = component
        self.status = status
        self.path = path
        self.expectedSHA256 = expectedSHA256
        self.actualSHA256 = actualSHA256
    }
}

public struct ModelArtifactVerificationReport: Codable, Equatable, Sendable {
    public var findings: [ModelArtifactFinding]

    public init(findings: [ModelArtifactFinding]) {
        self.findings = findings
    }

    public var isReady: Bool {
        !findings.isEmpty && findings.allSatisfy { $0.status == .verified }
    }

    public func finding(for component: ModelArtifactComponent) -> ModelArtifactFinding? {
        findings.first { $0.component == component }
    }
}

public struct ModelArtifactVerifier: Sendable {
    public init() {}

    public func verify(
        artifact: SelectedModelArtifact,
        relativeTo baseURL: URL
    ) -> ModelArtifactVerificationReport {
        let checks = [
            ArtifactCheck(
                component: .prefill,
                path: artifact.prefillPath,
                expectedSHA256: artifact.prefillSHA256
            ),
            ArtifactCheck(
                component: .decode,
                path: artifact.decodePath,
                expectedSHA256: artifact.decodeSHA256
            ),
            ArtifactCheck(
                component: .tokenizer,
                path: artifact.tokenizerPath,
                expectedSHA256: artifact.tokenizerSHA256
            )
        ]

        return ModelArtifactVerificationReport(
            findings: checks.map { verify(check: $0, relativeTo: baseURL) }
        )
    }

    private func verify(check: ArtifactCheck, relativeTo baseURL: URL) -> ModelArtifactFinding {
        guard let path = check.path, !path.isEmpty else {
            return ModelArtifactFinding(
                component: check.component,
                status: .missing,
                path: "",
                expectedSHA256: check.expectedSHA256,
                actualSHA256: nil
            )
        }

        let url = baseURL.appending(path: path)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return ModelArtifactFinding(
                component: check.component,
                status: .missing,
                path: path,
                expectedSHA256: check.expectedSHA256,
                actualSHA256: nil
            )
        }

        guard let expectedSHA256 = check.expectedSHA256, !expectedSHA256.isEmpty else {
            return ModelArtifactFinding(
                component: check.component,
                status: .hashUnavailable,
                path: path,
                expectedSHA256: nil,
                actualSHA256: nil
            )
        }

        let actualSHA256: String
        do {
            actualSHA256 = try ArtifactDigest.sha256Hex(for: url)
        } catch {
            return ModelArtifactFinding(
                component: check.component,
                status: .hashUnavailable,
                path: path,
                expectedSHA256: expectedSHA256,
                actualSHA256: nil
            )
        }

        return ModelArtifactFinding(
            component: check.component,
            status: actualSHA256 == expectedSHA256 ? .verified : .hashMismatch,
            path: path,
            expectedSHA256: expectedSHA256,
            actualSHA256: actualSHA256
        )
    }
}

private struct ArtifactCheck {
    var component: ModelArtifactComponent
    var path: String?
    var expectedSHA256: String?
}
