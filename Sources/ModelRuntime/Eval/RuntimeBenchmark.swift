import Foundation

#if canImport(Darwin)
import Darwin
#endif

public enum RuntimeThermalState: String, Codable, Equatable, Sendable {
    case nominal
    case fair
    case serious
    case critical
    case unavailable
}

public struct RuntimeTelemetrySnapshot: Codable, Equatable, Sendable {
    public var thermalState: RuntimeThermalState
    public var residentMemoryMB: Double?

    public init(
        thermalState: RuntimeThermalState = .unavailable,
        residentMemoryMB: Double? = nil
    ) {
        self.thermalState = thermalState
        self.residentMemoryMB = residentMemoryMB
    }
}

public protocol RuntimeTelemetryProbe: Sendable {
    func snapshot() -> RuntimeTelemetrySnapshot
}

public struct RuntimeTelemetrySummary: Codable, Equatable, Sendable {
    public var snapshots: [RuntimeTelemetrySnapshot]
    public var peakResidentMemoryMB: Double?
    public var thermalStates: [RuntimeThermalState]

    public init(snapshots: [RuntimeTelemetrySnapshot] = []) {
        self.snapshots = snapshots
        self.peakResidentMemoryMB = snapshots.compactMap(\.residentMemoryMB).max()
        self.thermalStates = snapshots.map(\.thermalState)
    }
}

public struct ProcessRuntimeTelemetryProbe: RuntimeTelemetryProbe {
    public init() {}

    public func snapshot() -> RuntimeTelemetrySnapshot {
        RuntimeTelemetrySnapshot(
            thermalState: currentRuntimeThermalState(),
            residentMemoryMB: currentResidentMemoryMB()
        )
    }
}

public struct RuntimeBenchmarkPrompt: Codable, Equatable, Sendable {
    public var id: String
    public var category: String
    public var language: String
    public var input: String
    public var maxNewTokens: Int
    public var qualityChecks: [String]
    public var qualityReference: RuntimeQualityReference?

    public init(
        id: String,
        category: String,
        language: String,
        input: String,
        maxNewTokens: Int,
        qualityChecks: [String] = [],
        qualityReference: RuntimeQualityReference? = nil
    ) {
        self.id = id
        self.category = category
        self.language = language
        self.input = input
        self.maxNewTokens = maxNewTokens
        self.qualityChecks = qualityChecks
        self.qualityReference = qualityReference
    }

    private enum CodingKeys: String, CodingKey {
        case id
        case category
        case language
        case input
        case maxNewTokens
        case qualityChecks
        case qualityReference
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        category = try container.decode(String.self, forKey: .category)
        language = try container.decode(String.self, forKey: .language)
        input = try container.decode(String.self, forKey: .input)
        maxNewTokens = try container.decode(Int.self, forKey: .maxNewTokens)
        qualityChecks = try container.decodeIfPresent([String].self, forKey: .qualityChecks) ?? []
        qualityReference = try container.decodeIfPresent(RuntimeQualityReference.self, forKey: .qualityReference)
    }
}

public enum RuntimeBenchmarkPromptSuiteError: Error, Equatable, Sendable {
    case invalidPrompts([String])
    case invalidQualityReferences([String])
}

public struct RuntimeBenchmarkPromptSuite: Codable, Equatable, Sendable {
    public static let requiredCategories = [
        "zh_short_instruction",
        "en_short_instruction",
        "code_small_fix",
        "watch_utility",
        "safety_refusal"
    ]
    public static let supportedCategories = requiredCategories + [
        "stop_sequence"
    ]

    private static let supportedSchemaVersion = 1
    private static let maxSmokePromptTokens = 256
    private static let minNewTokens = 16
    private static let maxNewTokens = 96

    public var schemaVersion: Int
    public var prompts: [RuntimeBenchmarkPrompt]

    public init(
        schemaVersion: Int,
        prompts: [RuntimeBenchmarkPrompt]
    ) {
        self.schemaVersion = schemaVersion
        self.prompts = prompts
    }

    public static func load(from url: URL) throws -> RuntimeBenchmarkPromptSuite {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        if let suite = try? decoder.decode(RuntimeBenchmarkPromptSuite.self, from: data) {
            return suite
        }

        let prompts = try decoder.decode([RuntimeBenchmarkPrompt].self, from: data)
        return RuntimeBenchmarkPromptSuite(schemaVersion: supportedSchemaVersion, prompts: prompts)
    }

    public var validationErrors: [String] {
        var errors: [String] = []

        if schemaVersion != Self.supportedSchemaVersion {
            errors.append("schemaVersion must be \(Self.supportedSchemaVersion)")
        }

        if prompts.isEmpty {
            errors.append("prompts must be a non-empty array")
        }

        var seenIDs = Set<String>()
        var seenCategories = Set<String>()
        for (index, prompt) in prompts.enumerated() {
            let trimmedID = prompt.id.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmedID.isEmpty {
                errors.append("prompt[\(index)].id must be a non-empty string")
            } else if seenIDs.contains(trimmedID) {
                errors.append("prompt[\(index)].id must be unique")
            } else {
                seenIDs.insert(trimmedID)
            }

            if Self.supportedCategories.contains(prompt.category) {
                seenCategories.insert(prompt.category)
            } else {
                errors.append("prompt[\(index)].category is unsupported")
            }

            if prompt.language.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                errors.append("prompt[\(index)].language must be a non-empty string")
            }

            if prompt.input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                errors.append("prompt[\(index)].input must be a non-empty string")
            } else if Self.estimatePromptTokens(prompt.input) > Self.maxSmokePromptTokens {
                errors.append("prompt[\(index)].input must fit the 256 token smoke baseline")
            }

            if prompt.maxNewTokens < Self.minNewTokens || prompt.maxNewTokens > Self.maxNewTokens {
                errors.append("prompt[\(index)].maxNewTokens must be between \(Self.minNewTokens) and \(Self.maxNewTokens)")
            }

            if prompt.qualityChecks.isEmpty || prompt.qualityChecks.contains(where: { $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }) {
                errors.append("prompt[\(index)].qualityChecks must be a non-empty array")
            }
        }

        for category in Self.requiredCategories where !seenCategories.contains(category) {
            errors.append("missing required category \(category)")
        }

        return errors
    }

    public func validatedPrompts() throws -> [RuntimeBenchmarkPrompt] {
        let errors = validationErrors
        guard errors.isEmpty else {
            throw RuntimeBenchmarkPromptSuiteError.invalidPrompts(errors)
        }
        return prompts
    }

    public func applyingQualityReferences(
        _ referenceSuite: RuntimeBenchmarkQualityReferenceSuite,
        requireAllPrompts: Bool = true
    ) throws -> RuntimeBenchmarkPromptSuite {
        let promptIDs = Set(prompts.map(\.id))
        var errors = referenceSuite.validationErrors(promptIDs: promptIDs)
        var referencesByPromptID: [String: RuntimeQualityReference] = [:]

        for reference in referenceSuite.references {
            let promptID = reference.promptID.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !promptID.isEmpty, promptIDs.contains(promptID), referencesByPromptID[promptID] == nil else {
                continue
            }

            referencesByPromptID[promptID] = RuntimeQualityReference(
                source: referenceSuite.source,
                tokenIDs: reference.tokenIDs
            )
        }

        if requireAllPrompts {
            for prompt in prompts where referencesByPromptID[prompt.id] == nil {
                errors.append("missing quality reference for \(prompt.id)")
            }
        }

        guard errors.isEmpty else {
            throw RuntimeBenchmarkPromptSuiteError.invalidQualityReferences(errors)
        }

        let referencedPrompts = prompts.map { prompt in
            var referencedPrompt = prompt
            referencedPrompt.qualityReference = referencesByPromptID[prompt.id]
            return referencedPrompt
        }
        return RuntimeBenchmarkPromptSuite(schemaVersion: schemaVersion, prompts: referencedPrompts)
    }

    private static func estimatePromptTokens(_ input: String) -> Int {
        Int(ceil(Double(Array(input).count) / 4.0))
    }
}

public struct RuntimeBenchmarkPromptQualityReference: Codable, Equatable, Sendable {
    public var promptID: String
    public var tokenIDs: [Int32]

    public init(promptID: String, tokenIDs: [Int32]) {
        self.promptID = promptID
        self.tokenIDs = tokenIDs
    }
}

public struct RuntimeBenchmarkQualityReferenceSuite: Codable, Equatable, Sendable {
    private static let supportedSchemaVersion = 1

    public var schemaVersion: Int
    public var source: String
    public var references: [RuntimeBenchmarkPromptQualityReference]

    public init(
        schemaVersion: Int,
        source: String,
        references: [RuntimeBenchmarkPromptQualityReference]
    ) {
        self.schemaVersion = schemaVersion
        self.source = source
        self.references = references
    }

    public static func load(from url: URL) throws -> RuntimeBenchmarkQualityReferenceSuite {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(RuntimeBenchmarkQualityReferenceSuite.self, from: data)
    }

    public var validationErrors: [String] {
        validationErrors(promptIDs: nil)
    }

    public func validationErrors(promptIDs: Set<String>?) -> [String] {
        var errors: [String] = []

        if schemaVersion != Self.supportedSchemaVersion {
            errors.append("qualityReferences.schemaVersion must be \(Self.supportedSchemaVersion)")
        }

        if source.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            errors.append("qualityReferences.source must be a non-empty string")
        }

        if references.isEmpty {
            errors.append("qualityReferences.references must be a non-empty array")
        }

        var seenPromptIDs = Set<String>()
        for (index, reference) in references.enumerated() {
            let promptID = reference.promptID.trimmingCharacters(in: .whitespacesAndNewlines)
            if promptID.isEmpty {
                errors.append("qualityReferences[\(index)].promptID must be a non-empty string")
            } else if seenPromptIDs.contains(promptID) {
                errors.append("qualityReferences[\(index)].promptID must be unique")
            } else {
                seenPromptIDs.insert(promptID)
            }

            if let promptIDs, !promptID.isEmpty, !promptIDs.contains(promptID) {
                errors.append("qualityReferences[\(index)].promptID \(promptID) does not exist in prompts")
            }

            if reference.tokenIDs.isEmpty {
                errors.append("qualityReferences[\(index)].tokenIDs must be a non-empty array")
            }
        }

        return errors
    }
}

public struct RuntimeQualityReference: Codable, Equatable, Sendable {
    public var source: String
    public var tokenIDs: [Int32]

    public init(source: String, tokenIDs: [Int32]) {
        self.source = source
        self.tokenIDs = tokenIDs
    }
}

public struct RuntimeQualityDrift: Codable, Equatable, Sendable {
    public var referenceSource: String
    public var comparedTokenCount: Int
    public var exactTokenMatchCount: Int
    public var tokenAgreement: Double
    public var firstMismatchIndex: Int?

    public init(
        referenceSource: String,
        comparedTokenCount: Int,
        exactTokenMatchCount: Int,
        tokenAgreement: Double,
        firstMismatchIndex: Int?
    ) {
        self.referenceSource = referenceSource
        self.comparedTokenCount = comparedTokenCount
        self.exactTokenMatchCount = exactTokenMatchCount
        self.tokenAgreement = tokenAgreement
        self.firstMismatchIndex = firstMismatchIndex
    }
}

public struct RuntimeBenchmarkArtifact: Codable, Equatable, Sendable {
    public var quantizationPolicyID: String
    public var graphInterface: String
    public var prefillModelPath: String
    public var decodeModelPath: String
    public var tokenizerPath: String?
    public var prefillSizeBytes: Int64?
    public var decodeSizeBytes: Int64?
    public var tokenizerSizeBytes: Int64?
    public var totalSizeBytes: Int64
    public var prefillSHA256: String?
    public var decodeSHA256: String?
    public var tokenizerSHA256: String?

    public init(
        quantizationPolicyID: String,
        graphInterface: String,
        prefillModelPath: String,
        decodeModelPath: String,
        tokenizerPath: String? = nil,
        prefillSizeBytes: Int64? = nil,
        decodeSizeBytes: Int64? = nil,
        tokenizerSizeBytes: Int64? = nil,
        prefillSHA256: String? = nil,
        decodeSHA256: String? = nil,
        tokenizerSHA256: String? = nil
    ) {
        self.quantizationPolicyID = quantizationPolicyID
        self.graphInterface = graphInterface
        self.prefillModelPath = prefillModelPath
        self.decodeModelPath = decodeModelPath
        self.tokenizerPath = tokenizerPath
        self.prefillSizeBytes = prefillSizeBytes
        self.decodeSizeBytes = decodeSizeBytes
        self.tokenizerSizeBytes = tokenizerSizeBytes
        totalSizeBytes = Self.uniqueArtifactByteCount(
            [
                (prefillModelPath, prefillSizeBytes),
                (decodeModelPath, decodeSizeBytes),
                (tokenizerPath, tokenizerSizeBytes),
            ]
        )
        self.prefillSHA256 = prefillSHA256
        self.decodeSHA256 = decodeSHA256
        self.tokenizerSHA256 = tokenizerSHA256
    }

    private static func uniqueArtifactByteCount(_ entries: [(String?, Int64?)]) -> Int64 {
        var seen = Set<String>()
        var total: Int64 = 0
        for (path, size) in entries {
            guard let path, let size else {
                continue
            }
            var key = path
            while key.hasSuffix("/") && key.count > 1 {
                key.removeLast()
            }
            guard seen.insert(key).inserted else {
                continue
            }
            total += size
        }
        return total
    }

    public init(
        selectedArtifact: SelectedModelArtifact,
        manifest: ModelManifest,
        assetBaseURL: URL,
        quantizationPolicyID: String
    ) throws {
        let prefillURL = assetBaseURL.appending(path: selectedArtifact.prefillPath)
        let decodeURL = assetBaseURL.appending(path: selectedArtifact.decodePath)
        let tokenizerURL = selectedArtifact.tokenizerPath.map { assetBaseURL.appending(path: $0) }

        try self.init(
            quantizationPolicyID: quantizationPolicyID,
            graphInterface: manifest.runtime.graphSchema.interface,
            prefillModelPath: selectedArtifact.prefillPath,
            decodeModelPath: selectedArtifact.decodePath,
            tokenizerPath: selectedArtifact.tokenizerPath,
            prefillSizeBytes: Self.byteCount(at: prefillURL),
            decodeSizeBytes: Self.byteCount(at: decodeURL),
            tokenizerSizeBytes: tokenizerURL.map { try Self.byteCount(at: $0) },
            prefillSHA256: selectedArtifact.prefillSHA256,
            decodeSHA256: selectedArtifact.decodeSHA256,
            tokenizerSHA256: selectedArtifact.tokenizerSHA256
        )
    }

    private static func byteCount(at url: URL) throws -> Int64 {
        let resolvedURL = url.resolvingSymlinksInPath()
        let values = try resolvedURL.resourceValues(forKeys: [.isDirectoryKey, .isRegularFileKey, .fileSizeKey])
        if values.isDirectory == true {
            return try directoryByteCount(at: resolvedURL)
        }

        if values.isRegularFile == true {
            return Int64(values.fileSize ?? 0)
        }

        return 0
    }

    private static func directoryByteCount(at directoryURL: URL) throws -> Int64 {
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

public struct RuntimeBenchmarkConfiguration: Codable, Equatable, Sendable {
    public var id: String
    public var sourceModelId: String
    public var runtime: String
    public var deviceProfile: DeviceProfile
    public var contextVariant: Int
    public var artifact: RuntimeBenchmarkArtifact?

    public init(
        id: String,
        sourceModelId: String,
        runtime: String,
        deviceProfile: DeviceProfile,
        contextVariant: Int,
        artifact: RuntimeBenchmarkArtifact? = nil
    ) {
        self.id = id
        self.sourceModelId = sourceModelId
        self.runtime = runtime
        self.deviceProfile = deviceProfile
        self.contextVariant = contextVariant
        self.artifact = artifact
    }
}

public struct RuntimeBenchmarkPromptResult: Codable, Equatable, Sendable {
    public var promptID: String
    public var category: String
    public var language: String
    public var maxNewTokens: Int
    public var text: String
    public var generatedTokenIDs: [Int32]
    public var generatedTokenCount: Int
    public var streamedTokenCount: Int
    public var timing: RuntimeTiming
    public var metrics: InferenceMetrics
    public var quality: RuntimeQualityDrift?
    public var terminationReason: InferenceTerminationReason?
    public var errorMessage: String?

    public init(
        promptID: String,
        category: String,
        language: String,
        maxNewTokens: Int,
        text: String = "",
        generatedTokenIDs: [Int32] = [],
        generatedTokenCount: Int = 0,
        streamedTokenCount: Int = 0,
        timing: RuntimeTiming = RuntimeTiming(),
        metrics: InferenceMetrics = InferenceMetrics(),
        quality: RuntimeQualityDrift? = nil,
        terminationReason: InferenceTerminationReason? = nil,
        errorMessage: String? = nil
    ) {
        self.promptID = promptID
        self.category = category
        self.language = language
        self.maxNewTokens = maxNewTokens
        self.text = text
        self.generatedTokenIDs = generatedTokenIDs
        self.generatedTokenCount = generatedTokenCount
        self.streamedTokenCount = streamedTokenCount
        self.timing = timing
        self.metrics = metrics
        self.quality = quality
        self.terminationReason = terminationReason
        self.errorMessage = errorMessage
    }

    public var succeeded: Bool {
        errorMessage == nil
    }

    public var decodeTokensPerSecond: Double {
        timing.decodeTokensPerSecond
    }
}

public struct RuntimeBenchmarkSummary: Codable, Equatable, Sendable {
    public var promptCount: Int
    public var succeededPromptCount: Int
    public var failedPromptCount: Int
    public var totalGeneratedTokens: Int
    public var averageFirstTokenMs: Double
    public var averageDecodeTokensPerSecond: Double
    public var averageTokenAgreement: Double?
    public var peakResidentMemoryMB: Double?
    public var thermalStates: [RuntimeThermalState]

    public init(
        promptCount: Int,
        succeededPromptCount: Int,
        failedPromptCount: Int,
        totalGeneratedTokens: Int,
        averageFirstTokenMs: Double,
        averageDecodeTokensPerSecond: Double,
        averageTokenAgreement: Double? = nil,
        peakResidentMemoryMB: Double? = nil,
        thermalStates: [RuntimeThermalState] = []
    ) {
        self.promptCount = promptCount
        self.succeededPromptCount = succeededPromptCount
        self.failedPromptCount = failedPromptCount
        self.totalGeneratedTokens = totalGeneratedTokens
        self.averageFirstTokenMs = averageFirstTokenMs
        self.averageDecodeTokensPerSecond = averageDecodeTokensPerSecond
        self.averageTokenAgreement = averageTokenAgreement
        self.peakResidentMemoryMB = peakResidentMemoryMB
        self.thermalStates = thermalStates
    }

    public var allPromptsSucceeded: Bool {
        failedPromptCount == 0
    }
}

public struct RuntimeBenchmarkGateTargets: Codable, Equatable, Sendable {
    public var maxFirstTokenMs: Double
    public var minDecodeTokensPerSecond: Double
    public var minAverageTokenAgreement: Double?
    public var maxPeakResidentMemoryMB: Double?
    public var disallowedThermalStates: [RuntimeThermalState]

    public init(
        maxFirstTokenMs: Double,
        minDecodeTokensPerSecond: Double,
        minAverageTokenAgreement: Double? = nil,
        maxPeakResidentMemoryMB: Double? = nil,
        disallowedThermalStates: [RuntimeThermalState] = [.critical]
    ) {
        self.maxFirstTokenMs = maxFirstTokenMs
        self.minDecodeTokensPerSecond = minDecodeTokensPerSecond
        self.minAverageTokenAgreement = minAverageTokenAgreement
        self.maxPeakResidentMemoryMB = maxPeakResidentMemoryMB
        self.disallowedThermalStates = disallowedThermalStates
    }

    public static func defaults(for deviceProfile: DeviceProfile) -> RuntimeBenchmarkGateTargets {
        switch deviceProfile {
        case .watchSE2:
            return RuntimeBenchmarkGateTargets(
                maxFirstTokenMs: 5_000,
                minDecodeTokensPerSecond: 1.5,
                minAverageTokenAgreement: 0.7
            )
        case .watchSE3:
            return RuntimeBenchmarkGateTargets(
                maxFirstTokenMs: 3_000,
                minDecodeTokensPerSecond: 3,
                minAverageTokenAgreement: 0.8
            )
        }
    }

    public func with(
        maxPeakResidentMemoryMB: Double? = nil,
        minAverageTokenAgreement: Double? = nil,
        disallowedThermalStates: [RuntimeThermalState]? = nil
    ) -> RuntimeBenchmarkGateTargets {
        RuntimeBenchmarkGateTargets(
            maxFirstTokenMs: maxFirstTokenMs,
            minDecodeTokensPerSecond: minDecodeTokensPerSecond,
            minAverageTokenAgreement: minAverageTokenAgreement ?? self.minAverageTokenAgreement,
            maxPeakResidentMemoryMB: maxPeakResidentMemoryMB ?? self.maxPeakResidentMemoryMB,
            disallowedThermalStates: disallowedThermalStates ?? self.disallowedThermalStates
        )
    }
}

public struct RuntimeBenchmarkGateMetrics: Codable, Equatable, Sendable {
    public var averageFirstTokenMs: Double
    public var averageDecodeTokensPerSecond: Double
    public var averageTokenAgreement: Double?
    public var peakResidentMemoryMB: Double?
    public var thermalStates: [RuntimeThermalState]

    public init(
        averageFirstTokenMs: Double,
        averageDecodeTokensPerSecond: Double,
        averageTokenAgreement: Double?,
        peakResidentMemoryMB: Double?,
        thermalStates: [RuntimeThermalState]
    ) {
        self.averageFirstTokenMs = averageFirstTokenMs
        self.averageDecodeTokensPerSecond = averageDecodeTokensPerSecond
        self.averageTokenAgreement = averageTokenAgreement
        self.peakResidentMemoryMB = peakResidentMemoryMB
        self.thermalStates = thermalStates
    }
}

public struct RuntimeBenchmarkGateResult: Codable, Equatable, Sendable {
    public var ok: Bool
    public var failures: [String]
    public var targets: RuntimeBenchmarkGateTargets
    public var metrics: RuntimeBenchmarkGateMetrics

    public init(
        ok: Bool,
        failures: [String],
        targets: RuntimeBenchmarkGateTargets,
        metrics: RuntimeBenchmarkGateMetrics
    ) {
        self.ok = ok
        self.failures = failures
        self.targets = targets
        self.metrics = metrics
    }
}

public enum RuntimeBenchmarkGate {
    public static func evaluate(
        _ report: RuntimeBenchmarkReport,
        targets: RuntimeBenchmarkGateTargets? = nil
    ) -> RuntimeBenchmarkGateResult {
        let resolvedTargets = targets ?? RuntimeBenchmarkGateTargets.defaults(for: report.configuration.deviceProfile)
        let metrics = RuntimeBenchmarkGateMetrics(
            averageFirstTokenMs: report.summary.averageFirstTokenMs,
            averageDecodeTokensPerSecond: report.summary.averageDecodeTokensPerSecond,
            averageTokenAgreement: report.summary.averageTokenAgreement,
            peakResidentMemoryMB: report.summary.peakResidentMemoryMB,
            thermalStates: report.summary.thermalStates
        )
        var failures: [String] = []

        if report.summary.failedPromptCount > 0 {
            failures.append("\(report.summary.failedPromptCount) prompts failed")
        }

        if metrics.averageFirstTokenMs > resolvedTargets.maxFirstTokenMs {
            failures.append("first token \(metrics.averageFirstTokenMs)ms exceeds \(resolvedTargets.maxFirstTokenMs)ms target")
        }

        if metrics.averageDecodeTokensPerSecond < resolvedTargets.minDecodeTokensPerSecond {
            failures.append("decode \(metrics.averageDecodeTokensPerSecond) tok/s is below \(resolvedTargets.minDecodeTokensPerSecond) tok/s target")
        }

        if let minAgreement = resolvedTargets.minAverageTokenAgreement {
            if let agreement = metrics.averageTokenAgreement {
                if agreement < minAgreement {
                    failures.append("quality agreement \(agreement) is below \(minAgreement) target")
                }
            } else {
                failures.append("quality agreement is unavailable")
            }
        }

        if let maxMemory = resolvedTargets.maxPeakResidentMemoryMB {
            if let peakMemory = metrics.peakResidentMemoryMB {
                if peakMemory > maxMemory {
                    failures.append("peak resident memory \(peakMemory)MB exceeds \(maxMemory)MB target")
                }
            } else {
                failures.append("peak resident memory is unavailable")
            }
        }

        let disallowedThermalStates = metrics.thermalStates.filter {
            resolvedTargets.disallowedThermalStates.contains($0)
        }
        if !disallowedThermalStates.isEmpty {
            let names = Array(Set(disallowedThermalStates.map(\.rawValue))).sorted().joined(separator: ", ")
            failures.append("thermal states include \(names)")
        }

        return RuntimeBenchmarkGateResult(
            ok: failures.isEmpty,
            failures: failures,
            targets: resolvedTargets,
            metrics: metrics
        )
    }
}

public struct RuntimeBenchmarkReport: Codable, Equatable, Sendable {
    public var configuration: RuntimeBenchmarkConfiguration
    public var loadTiming: RuntimeTiming
    public var promptResults: [RuntimeBenchmarkPromptResult]
    public var telemetry: RuntimeTelemetrySummary
    public var summary: RuntimeBenchmarkSummary

    public init(
        configuration: RuntimeBenchmarkConfiguration,
        loadTiming: RuntimeTiming,
        promptResults: [RuntimeBenchmarkPromptResult],
        telemetry: RuntimeTelemetrySummary = RuntimeTelemetrySummary(),
        summary: RuntimeBenchmarkSummary
    ) {
        self.configuration = configuration
        self.loadTiming = loadTiming
        self.promptResults = promptResults
        self.telemetry = telemetry
        self.summary = summary
    }
}

public struct RuntimeBenchmarkRunner: Sendable {
    public init() {}

    public func run(
        runtime: any InferenceRuntime,
        configuration: RuntimeBenchmarkConfiguration,
        prompts: [RuntimeBenchmarkPrompt],
        shouldCancel: @escaping @Sendable () -> Bool = { false },
        telemetryProbe: any RuntimeTelemetryProbe = ProcessRuntimeTelemetryProbe()
    ) async throws -> RuntimeBenchmarkReport {
        var telemetrySnapshots = [telemetryProbe.snapshot()]
        let loadResult: Result<RuntimeTiming, Error>
        do {
            loadResult = .success(try await runtime.load())
        } catch {
            loadResult = .failure(error)
        }
        telemetrySnapshots.append(telemetryProbe.snapshot())

        let loadTiming = (try? loadResult.get()) ?? RuntimeTiming()
        var promptResults: [RuntimeBenchmarkPromptResult]
        switch loadResult {
        case .success:
            promptResults = []
            promptResults.reserveCapacity(prompts.count)
            for prompt in prompts {
                let result = await runPrompt(
                    prompt,
                    runtime: runtime,
                    shouldCancel: shouldCancel
                )
                promptResults.append(result)
                telemetrySnapshots.append(telemetryProbe.snapshot())
            }
        case .failure(let error):
            let message = runtimeBenchmarkErrorMessage(error)
            promptResults = prompts.map { prompt in
                RuntimeBenchmarkPromptResult(
                    promptID: prompt.id,
                    category: prompt.category,
                    language: prompt.language,
                    maxNewTokens: prompt.maxNewTokens,
                    errorMessage: message
                )
            }
        }
        let telemetry = RuntimeTelemetrySummary(snapshots: telemetrySnapshots)

        return RuntimeBenchmarkReport(
            configuration: configuration,
            loadTiming: loadTiming,
            promptResults: promptResults,
            telemetry: telemetry,
            summary: RuntimeBenchmarkSummary(results: promptResults, telemetry: telemetry)
        )
    }

    private func runPrompt(
        _ prompt: RuntimeBenchmarkPrompt,
        runtime: any InferenceRuntime,
        shouldCancel: @escaping @Sendable () -> Bool
    ) async -> RuntimeBenchmarkPromptResult {
        do {
            let request = InferenceRequest(prompt: prompt.input, maxNewTokens: prompt.maxNewTokens)
            let run = try await runGeneration(request: request, runtime: runtime, shouldCancel: shouldCancel)
            return RuntimeBenchmarkPromptResult(
                promptID: prompt.id,
                category: prompt.category,
                language: prompt.language,
                maxNewTokens: prompt.maxNewTokens,
                text: run.result.text,
                generatedTokenIDs: run.result.generatedTokenIDs,
                generatedTokenCount: run.result.tokens.count,
                streamedTokenCount: run.streamedTokenCount,
                timing: run.result.timing,
                metrics: run.result.metrics,
                quality: qualityDrift(
                    reference: prompt.qualityReference,
                    generatedTokenIDs: run.result.generatedTokenIDs
                ),
                terminationReason: run.result.terminationReason
            )
        } catch {
            return RuntimeBenchmarkPromptResult(
                promptID: prompt.id,
                category: prompt.category,
                language: prompt.language,
                maxNewTokens: prompt.maxNewTokens,
                errorMessage: runtimeBenchmarkErrorMessage(error)
            )
        }
    }

    private func runGeneration(
        request: InferenceRequest,
        runtime: any InferenceRuntime,
        shouldCancel: @escaping @Sendable () -> Bool
    ) async throws -> (result: InferenceResult, streamedTokenCount: Int) {
        if let streamingRuntime = runtime as? any StreamingInferenceRuntime {
            var streamedTokenCount = 0
            var completedResult: InferenceResult?

            for try await event in streamingRuntime.stream(request: request, shouldCancel: shouldCancel) {
                switch event {
                case .token:
                    streamedTokenCount += 1
                case .completed(let result):
                    completedResult = result
                }
            }

            guard let completedResult else {
                throw InferenceRuntimeError.predictionFailed(message: "Streaming runtime finished without a completion event.")
            }
            return (completedResult, streamedTokenCount)
        }

        return (try await runtime.generate(request: request, shouldCancel: shouldCancel), 0)
    }
}

private extension RuntimeBenchmarkSummary {
    init(results: [RuntimeBenchmarkPromptResult], telemetry: RuntimeTelemetrySummary) {
        let successes = results.filter(\.succeeded)
        self.init(
            promptCount: results.count,
            succeededPromptCount: successes.count,
            failedPromptCount: results.count - successes.count,
            totalGeneratedTokens: successes.reduce(0) { $0 + $1.generatedTokenCount },
            averageFirstTokenMs: roundedBenchmarkAverage(successes.map(\.timing.firstTokenMs)),
            averageDecodeTokensPerSecond: roundedBenchmarkAverage(successes.map(\.decodeTokensPerSecond)),
            averageTokenAgreement: roundedBenchmarkOptionalAverage(successes.compactMap(\.quality?.tokenAgreement)),
            peakResidentMemoryMB: telemetry.peakResidentMemoryMB,
            thermalStates: telemetry.thermalStates
        )
    }
}

private func qualityDrift(
    reference: RuntimeQualityReference?,
    generatedTokenIDs: [Int32]
) -> RuntimeQualityDrift? {
    guard let reference else {
        return nil
    }

    let comparedTokenCount = min(reference.tokenIDs.count, generatedTokenIDs.count)
    var exactTokenMatchCount = 0
    var firstMismatchIndex: Int?
    for index in 0..<comparedTokenCount {
        if reference.tokenIDs[index] == generatedTokenIDs[index] {
            exactTokenMatchCount += 1
        } else if firstMismatchIndex == nil {
            firstMismatchIndex = index
        }
    }

    if firstMismatchIndex == nil && reference.tokenIDs.count != generatedTokenIDs.count {
        firstMismatchIndex = comparedTokenCount
    }

    let denominator = max(reference.tokenIDs.count, generatedTokenIDs.count)
    let tokenAgreement = denominator == 0
        ? 1.0
        : roundedBenchmarkRatio(Double(exactTokenMatchCount) / Double(denominator))

    return RuntimeQualityDrift(
        referenceSource: reference.source,
        comparedTokenCount: comparedTokenCount,
        exactTokenMatchCount: exactTokenMatchCount,
        tokenAgreement: tokenAgreement,
        firstMismatchIndex: firstMismatchIndex
    )
}

private func roundedBenchmarkAverage(_ values: [Double]) -> Double {
    guard !values.isEmpty else {
        return 0
    }

    let average = values.reduce(0, +) / Double(values.count)
    return roundedBenchmarkRatio(average)
}

private func roundedBenchmarkOptionalAverage(_ values: [Double]) -> Double? {
    guard !values.isEmpty else {
        return nil
    }

    return roundedBenchmarkAverage(values)
}

private func roundedBenchmarkRatio(_ value: Double) -> Double {
    (value * 100).rounded() / 100
}

private func runtimeBenchmarkErrorMessage(_ error: Error) -> String {
    if let runtimeError = error as? InferenceRuntimeError {
        return runtimeError.userMessage
    }

    return String(describing: error)
}

private func currentRuntimeThermalState() -> RuntimeThermalState {
    #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
    if #available(macOS 10.10, iOS 11.0, watchOS 4.0, tvOS 11.0, *) {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal:
            return .nominal
        case .fair:
            return .fair
        case .serious:
            return .serious
        case .critical:
            return .critical
        @unknown default:
            return .unavailable
        }
    }
    #endif

    return .unavailable
}

private func currentResidentMemoryMB() -> Double? {
    #if canImport(Darwin)
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
    let result = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { reboundPointer in
            task_info(
                mach_task_self_,
                task_flavor_t(MACH_TASK_BASIC_INFO),
                reboundPointer,
                &count
            )
        }
    }

    guard result == KERN_SUCCESS else {
        return nil
    }

    let megabytes = Double(info.resident_size) / 1_048_576
    return (megabytes * 100).rounded() / 100
    #else
    return nil
    #endif
}
