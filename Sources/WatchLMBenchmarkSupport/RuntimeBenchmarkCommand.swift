import Foundation
import WatchLMCore

#if canImport(CoreML)
import CoreML
#endif

public enum RuntimeBenchmarkRuntime: String, Codable, Equatable, Sendable {
    case mock
    case coreML = "coreml"
}

public enum CoreMLLoadTarget: String, Codable, Equatable, Sendable {
    case both
    case prefill
    case decode
}

public enum CoreMLBenchmarkGraphInterface: String, Codable, Equatable, Sendable {
    case explicitKV = "logits-layered-kv"
    case statefulKV = "stateful-kv"
    case statefulStepKV = "stateful-step-kv"
}

public enum RuntimeBenchmarkCommandError: Error, Equatable, CustomStringConvertible, Sendable {
    case missingOption(String)
    case invalidOption(String)
    case unsupportedRuntime(String)

    public var description: String {
        switch self {
        case .missingOption(let option):
            "Missing required option: \(option)"
        case .invalidOption(let message):
            "Invalid option: \(message)"
        case .unsupportedRuntime(let message):
            "Unsupported runtime: \(message)"
        }
    }
}

public struct CoreMLDiagnosticPromptResult: Codable, Equatable, Sendable {
    public var promptID: String
    public var category: String
    public var language: String
    public var promptTokenIDs: [Int32]
    public var requestedPrefixTokenCount: Int?
    public var prefixTokenCount: Int
    public var sourcePromptTokenCount: Int?
    public var prefillTopK: [TokenLogit]
    public var decodeTopK: [TokenLogit]
    public var errorMessage: String?

    public init(
        promptID: String,
        category: String,
        language: String,
        promptTokenIDs: [Int32] = [],
        requestedPrefixTokenCount: Int? = nil,
        prefixTokenCount: Int? = nil,
        sourcePromptTokenCount: Int? = nil,
        prefillTopK: [TokenLogit] = [],
        decodeTopK: [TokenLogit] = [],
        errorMessage: String? = nil
    ) {
        self.promptID = promptID
        self.category = category
        self.language = language
        self.promptTokenIDs = promptTokenIDs
        self.requestedPrefixTokenCount = requestedPrefixTokenCount
        self.prefixTokenCount = prefixTokenCount ?? promptTokenIDs.count
        self.sourcePromptTokenCount = sourcePromptTokenCount
        self.prefillTopK = prefillTopK
        self.decodeTopK = decodeTopK
        self.errorMessage = errorMessage
    }

    public init(
        prompt: RuntimeBenchmarkPrompt,
        requestedPrefixTokenCount: Int? = nil,
        prefixTokenCount: Int? = nil,
        sourcePromptTokenCount: Int? = nil,
        result: Result<CoreMLPrefillDecodeDiagnosticReport, Error>
    ) {
        switch result {
        case .success(let report):
            self.init(
                promptID: prompt.id,
                category: prompt.category,
                language: prompt.language,
                promptTokenIDs: report.promptTokenIDs,
                requestedPrefixTokenCount: requestedPrefixTokenCount,
                prefixTokenCount: prefixTokenCount,
                sourcePromptTokenCount: sourcePromptTokenCount,
                prefillTopK: report.prefillTopK,
                decodeTopK: report.decodeTopK
            )
        case .failure(let error):
            self.init(
                promptID: prompt.id,
                category: prompt.category,
                language: prompt.language,
                requestedPrefixTokenCount: requestedPrefixTokenCount,
                prefixTokenCount: prefixTokenCount,
                sourcePromptTokenCount: sourcePromptTokenCount,
                errorMessage: String(describing: error)
            )
        }
    }

    public var prefillTokenID: Int32? {
        prefillTopK.first?.tokenID
    }

    public var firstDecodeTokenID: Int32? {
        decodeTopK.first?.tokenID
    }
}

public struct CoreMLDiagnosticsSummary: Codable, Equatable, Sendable {
    public var promptCount: Int
    public var succeededPromptCount: Int
    public var failedPromptCount: Int

    public init(promptResults: [CoreMLDiagnosticPromptResult]) {
        promptCount = promptResults.count
        failedPromptCount = promptResults.filter { $0.errorMessage != nil }.count
        succeededPromptCount = promptCount - failedPromptCount
    }
}

public struct CoreMLDiagnosticsReport: Codable, Equatable, Sendable {
    public var configuration: RuntimeBenchmarkConfiguration
    public var topK: Int
    public var promptResults: [CoreMLDiagnosticPromptResult]
    public var summary: CoreMLDiagnosticsSummary

    public init(
        configuration: RuntimeBenchmarkConfiguration,
        topK: Int,
        promptResults: [CoreMLDiagnosticPromptResult]
    ) {
        self.configuration = configuration
        self.topK = topK
        self.promptResults = promptResults
        self.summary = CoreMLDiagnosticsSummary(promptResults: promptResults)
    }
}

public struct RuntimeBenchmarkCommandOptions: Equatable, Sendable {
    public var runtime: RuntimeBenchmarkRuntime
    public var promptsURL: URL
    public var teacherReferencesURL: URL?
    public var outputURL: URL?
    public var promptIDs: [String]?
    public var promptLimit: Int?
    public var maxNewTokens: Int?
    public var requireAllReferences: Bool
    public var deviceProfile: DeviceProfile
    public var contextVariant: Int
    public var configurationID: String
    public var sourceModelID: String
    public var policyID: String
    public var prefillModelURL: URL?
    public var decodeModelURL: URL?
    public var tokenizerURL: URL?
    public var coreMLGraphInterface: CoreMLBenchmarkGraphInterface
    public var diagnosticsTopK: Int?
    public var diagnosticsPrefixLengths: [Int]?
    public var sensitivityBaselineURL: URL?
    public var sensitivityCandidateURL: URL?
    public var loadOnly: Bool
    public var coreMLLoadTarget: CoreMLLoadTarget
    public var mockTokens: [String]
    public var mockTokenIDs: [Int32]

    public init(
        runtime: RuntimeBenchmarkRuntime = .coreML,
        promptsURL: URL,
        teacherReferencesURL: URL? = nil,
        outputURL: URL? = nil,
        promptIDs: [String]? = nil,
        promptLimit: Int? = nil,
        maxNewTokens: Int? = nil,
        requireAllReferences: Bool = true,
        deviceProfile: DeviceProfile = .watchSE2,
        contextVariant: Int = 16,
        configurationID: String = "watchlm-benchmark",
        sourceModelID: String = "openbmb/MiniCPM5-1B",
        policyID: String = "manual",
        prefillModelURL: URL? = nil,
        decodeModelURL: URL? = nil,
        tokenizerURL: URL? = nil,
        coreMLGraphInterface: CoreMLBenchmarkGraphInterface = .explicitKV,
        diagnosticsTopK: Int? = nil,
        diagnosticsPrefixLengths: [Int]? = nil,
        sensitivityBaselineURL: URL? = nil,
        sensitivityCandidateURL: URL? = nil,
        loadOnly: Bool = false,
        coreMLLoadTarget: CoreMLLoadTarget = .both,
        mockTokens: [String] = ["A"],
        mockTokenIDs: [Int32] = [1]
    ) {
        self.runtime = runtime
        self.promptsURL = promptsURL
        self.teacherReferencesURL = teacherReferencesURL
        self.outputURL = outputURL
        self.promptIDs = promptIDs
        self.promptLimit = promptLimit
        self.maxNewTokens = maxNewTokens
        self.requireAllReferences = requireAllReferences
        self.deviceProfile = deviceProfile
        self.contextVariant = contextVariant
        self.configurationID = configurationID
        self.sourceModelID = sourceModelID
        self.policyID = policyID
        self.prefillModelURL = prefillModelURL
        self.decodeModelURL = decodeModelURL
        self.tokenizerURL = tokenizerURL
        self.coreMLGraphInterface = coreMLGraphInterface
        self.diagnosticsTopK = diagnosticsTopK
        self.diagnosticsPrefixLengths = diagnosticsPrefixLengths
        self.sensitivityBaselineURL = sensitivityBaselineURL
        self.sensitivityCandidateURL = sensitivityCandidateURL
        self.loadOnly = loadOnly
        self.coreMLLoadTarget = coreMLLoadTarget
        self.mockTokens = mockTokens
        self.mockTokenIDs = mockTokenIDs
    }

    public var runsSensitivityComparison: Bool {
        sensitivityBaselineURL != nil || sensitivityCandidateURL != nil
    }

    public static func parse(
        _ arguments: [String],
        currentDirectory: URL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    ) throws -> RuntimeBenchmarkCommandOptions {
        var values = ParsedBenchmarkArguments(currentDirectory: currentDirectory)
        var index = 0
        while index < arguments.count {
            let argument = arguments[index]
            switch argument {
            case "--runtime":
                values.runtime = try RuntimeBenchmarkRuntime(rawValue: value(after: argument, in: arguments, at: &index))
                    .orThrowInvalid("\(argument) must be mock or coreml")
            case "--prompts":
                values.promptsURL = values.resolve(try value(after: argument, in: arguments, at: &index))
            case "--teacher":
                values.teacherReferencesURL = values.resolve(try value(after: argument, in: arguments, at: &index))
            case "--output":
                values.outputURL = values.resolve(try value(after: argument, in: arguments, at: &index))
            case "--prompt-ids":
                values.promptIDs = try parseStringList(value(after: argument, in: arguments, at: &index), option: argument)
            case "--prompt-limit":
                values.promptLimit = try parsePositiveInt(value(after: argument, in: arguments, at: &index), option: argument)
            case "--max-new-tokens":
                values.maxNewTokens = try parsePositiveInt(value(after: argument, in: arguments, at: &index), option: argument)
            case "--allow-missing-references":
                values.requireAllReferences = false
            case "--device-profile":
                values.deviceProfile = try DeviceProfile(rawValue: value(after: argument, in: arguments, at: &index))
                    .orThrowInvalid("\(argument) must be watch-se-2 or watch-se-3")
            case "--context":
                values.contextVariant = try parsePositiveInt(value(after: argument, in: arguments, at: &index), option: argument)
            case "--id":
                values.configurationID = try value(after: argument, in: arguments, at: &index)
            case "--source-model":
                values.sourceModelID = try value(after: argument, in: arguments, at: &index)
            case "--policy-id":
                values.policyID = try value(after: argument, in: arguments, at: &index)
            case "--prefill":
                values.prefillModelURL = values.resolve(try value(after: argument, in: arguments, at: &index))
            case "--decode":
                values.decodeModelURL = values.resolve(try value(after: argument, in: arguments, at: &index))
            case "--tokenizer":
                values.tokenizerURL = values.resolve(try value(after: argument, in: arguments, at: &index))
            case "--coreml-graph-interface":
                values.coreMLGraphInterface = try CoreMLBenchmarkGraphInterface(
                    rawValue: value(after: argument, in: arguments, at: &index)
                ).orThrowInvalid("\(argument) must be logits-layered-kv, stateful-kv, or stateful-step-kv")
            case "--diagnostics-top-k":
                values.diagnosticsTopK = try parsePositiveInt(value(after: argument, in: arguments, at: &index), option: argument)
            case "--diagnostics-prefix-lengths":
                values.diagnosticsPrefixLengths = try parsePositiveIntList(
                    value(after: argument, in: arguments, at: &index),
                    option: argument
                )
            case "--sensitivity-baseline":
                values.sensitivityBaselineURL = values.resolve(try value(after: argument, in: arguments, at: &index))
            case "--sensitivity-candidate":
                values.sensitivityCandidateURL = values.resolve(try value(after: argument, in: arguments, at: &index))
            case "--load-only":
                values.loadOnly = true
            case "--coreml-load-target":
                values.coreMLLoadTarget = try CoreMLLoadTarget(rawValue: value(after: argument, in: arguments, at: &index))
                    .orThrowInvalid("\(argument) must be both, prefill, or decode")
            case "--mock-tokens":
                values.mockTokens = try parseStringList(value(after: argument, in: arguments, at: &index), option: argument)
            case "--mock-token-ids":
                values.mockTokenIDs = try parseInt32List(value(after: argument, in: arguments, at: &index), option: argument)
            case "--help", "-h":
                throw RuntimeBenchmarkCommandError.invalidOption(RuntimeBenchmarkCommand.usage)
            default:
                throw RuntimeBenchmarkCommandError.invalidOption("unknown argument \(argument)")
            }
            index += 1
        }
        return values.options()
    }
}

public struct RuntimeBenchmarkCommand: Sendable {
    public static let usage = """
    Usage:
      swift run WatchLMBenchmark --runtime coreml --prefill PATH --decode PATH --tokenizer PATH [options]
      swift run WatchLMBenchmark --runtime mock --mock-token-ids 10,11 --mock-tokens A,B [options]
      swift run WatchLMBenchmark --sensitivity-baseline FP16.json --sensitivity-candidate CANDIDATE.json [options]

    Options:
      --prompts PATH                 Prompt suite JSON. Defaults to tools/benchmark/fixtures/benchmark-prompts.json.
      --teacher PATH                 Teacher reference sidecar JSON.
      --output PATH                  Write RuntimeBenchmarkReport JSON to this path. Without it, JSON is printed to stdout.
      --prompt-ids A,B               Run specific prompt ids in the requested order.
      --prompt-limit N               Run only the first N prompts.
      --max-new-tokens N             Cap each prompt's maxNewTokens for smoke runs.
      --allow-missing-references     Do not require every selected prompt to have teacher tokens.
      --device-profile watch-se-2|watch-se-3
      --context N
      --policy-id ID
      --id ID
      --coreml-graph-interface logits-layered-kv|stateful-kv|stateful-step-kv
      --diagnostics-top-k N        Run Core ML logits diagnostics instead of generation.
      --diagnostics-prefix-lengths 1,2,4
                                   Run diagnostics on token prefixes.
      --sensitivity-baseline PATH  Baseline Core ML diagnostics JSON for quantization drift scoring.
      --sensitivity-candidate PATH Candidate Core ML diagnostics JSON for quantization drift scoring.
      --load-only                  Load runtime artifacts and skip prompt generation.
      --coreml-load-target both|prefill|decode
    """

    private let options: RuntimeBenchmarkCommandOptions

    public init(options: RuntimeBenchmarkCommandOptions) {
        self.options = options
    }

    public func run() async throws -> RuntimeBenchmarkReport {
        let prompts = options.loadOnly ? [] : try loadPrompts()
        let runtime = try makeRuntime()
        let report = try await RuntimeBenchmarkRunner().run(
            runtime: runtime,
            configuration: makeConfiguration(),
            prompts: prompts
        )

        if let outputURL = options.outputURL {
            try write(report: report, to: outputURL)
        }
        return report
    }

    public func runDiagnostics() throws -> CoreMLDiagnosticsReport {
        #if canImport(CoreML)
        guard options.runtime == .coreML else {
            throw RuntimeBenchmarkCommandError.unsupportedRuntime("Core ML diagnostics require --runtime coreml")
        }
        guard let topK = options.diagnosticsTopK else {
            throw RuntimeBenchmarkCommandError.missingOption("--diagnostics-top-k")
        }

        let prompts = try loadPrompts()
        let tokenizer = try makeCoreMLTokenizer()
        let diagnostics = CoreMLPrefillDecodeDiagnostics(
            bundle: try makeCoreMLBundle(),
            tokenizer: tokenizer
        )
        let results = try prompts.flatMap { prompt in
            try diagnosticResults(
                for: prompt,
                diagnostics: diagnostics,
                tokenizer: tokenizer,
                topK: topK
            )
        }
        let report = CoreMLDiagnosticsReport(
            configuration: makeConfiguration(),
            topK: topK,
            promptResults: results
        )

        if let outputURL = options.outputURL {
            try write(diagnosticsReport: report, to: outputURL)
        }
        return report
        #else
        throw RuntimeBenchmarkCommandError.unsupportedRuntime("Core ML is unavailable on this platform")
        #endif
    }

    public func runSensitivityComparison() throws -> QuantizationSensitivityReport {
        let baselineURL = try requiredURL(options.sensitivityBaselineURL, "--sensitivity-baseline")
        let candidateURL = try requiredURL(options.sensitivityCandidateURL, "--sensitivity-candidate")
        let decoder = JSONDecoder()
        let baselineReport = try decoder.decode(
            CoreMLDiagnosticsReport.self,
            from: Data(contentsOf: baselineURL)
        )
        let candidateReport = try decoder.decode(
            CoreMLDiagnosticsReport.self,
            from: Data(contentsOf: candidateURL)
        )
        let report = try QuantizationSensitivityScorer.compare(
            baselinePolicyID: Self.sensitivityPolicyID(from: baselineReport),
            candidatePolicyID: Self.sensitivityPolicyID(from: candidateReport),
            baseline: Self.diagnosticPoints(from: baselineReport),
            candidate: Self.diagnosticPoints(from: candidateReport)
        )

        if let outputURL = options.outputURL {
            try write(sensitivityReport: report, to: outputURL)
        }
        return report
    }

    public static func encode(report: RuntimeBenchmarkReport) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(report)
    }

    public static func encode(diagnosticsReport: CoreMLDiagnosticsReport) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(diagnosticsReport)
    }

    public static func encode(sensitivityReport: QuantizationSensitivityReport) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(sensitivityReport)
    }

    private static func diagnosticPoints(from report: CoreMLDiagnosticsReport) -> [LogitsDiagnosticPoint] {
        report.promptResults.compactMap { result in
            guard result.errorMessage == nil else {
                return nil
            }
            return LogitsDiagnosticPoint(
                promptID: result.promptID,
                category: result.category,
                language: result.language,
                prefixTokenCount: result.prefixTokenCount,
                prefillTopK: result.prefillTopK,
                decodeTopK: result.decodeTopK
            )
        }
    }

    private static func sensitivityPolicyID(from report: CoreMLDiagnosticsReport) -> String {
        report.configuration.artifact?.quantizationPolicyID ?? report.configuration.id
    }

    private func loadPrompts() throws -> [RuntimeBenchmarkPrompt] {
        var suite = try RuntimeBenchmarkPromptSuite.load(from: options.promptsURL)
        _ = try suite.validatedPrompts()

        if let promptIDs = options.promptIDs {
            let promptsByID = Dictionary(uniqueKeysWithValues: suite.prompts.map { prompt in
                (prompt.id, prompt)
            })
            let selectedPrompts = try promptIDs.map { promptID in
                guard let prompt = promptsByID[promptID] else {
                    throw RuntimeBenchmarkCommandError.invalidOption("unknown prompt id \(promptID)")
                }
                return prompt
            }
            suite = RuntimeBenchmarkPromptSuite(
                schemaVersion: suite.schemaVersion,
                prompts: selectedPrompts
            )
        }

        if let promptLimit = options.promptLimit {
            suite = RuntimeBenchmarkPromptSuite(
                schemaVersion: suite.schemaVersion,
                prompts: Array(suite.prompts.prefix(promptLimit))
            )
        }

        if let maxNewTokens = options.maxNewTokens {
            suite = RuntimeBenchmarkPromptSuite(
                schemaVersion: suite.schemaVersion,
                prompts: suite.prompts.map { prompt in
                    var cappedPrompt = prompt
                    cappedPrompt.maxNewTokens = min(cappedPrompt.maxNewTokens, maxNewTokens)
                    return cappedPrompt
                }
            )
        }

        if let teacherReferencesURL = options.teacherReferencesURL {
            let references = try RuntimeBenchmarkQualityReferenceSuite.load(from: teacherReferencesURL)
            let maxNewTokensByPromptID = Dictionary(uniqueKeysWithValues: suite.prompts.map { prompt in
                (prompt.id, prompt.maxNewTokens)
            })
            let selectedReferences = RuntimeBenchmarkQualityReferenceSuite(
                schemaVersion: references.schemaVersion,
                source: references.source,
                references: references.references.compactMap { reference in
                    let promptID = reference.promptID.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard let maxNewTokens = maxNewTokensByPromptID[promptID] else {
                        return nil
                    }
                    return RuntimeBenchmarkPromptQualityReference(
                        promptID: reference.promptID,
                        tokenIDs: Array(reference.tokenIDs.prefix(maxNewTokens))
                    )
                }
            )
            suite = try suite.applyingQualityReferences(
                selectedReferences,
                requireAllPrompts: options.requireAllReferences
            )
        }
        return suite.prompts
    }

    #if canImport(CoreML)
    private func diagnosticResults(
        for prompt: RuntimeBenchmarkPrompt,
        diagnostics: CoreMLPrefillDecodeDiagnostics,
        tokenizer: any TextTokenizer,
        topK: Int
    ) throws -> [CoreMLDiagnosticPromptResult] {
        guard let prefixLengths = options.diagnosticsPrefixLengths else {
            return [
                CoreMLDiagnosticPromptResult(
                    prompt: prompt,
                    result: Result {
                        try diagnostics.run(prompt: prompt.input, topK: topK)
                    }
                )
            ]
        }

        let sourceTokenIDs = try tokenizer.encode(prompt.input)
        return prefixLengths.map { requestedLength in
            let prefixTokenIDs = Array(sourceTokenIDs.prefix(requestedLength))
            return CoreMLDiagnosticPromptResult(
                prompt: prompt,
                requestedPrefixTokenCount: requestedLength,
                prefixTokenCount: prefixTokenIDs.count,
                sourcePromptTokenCount: sourceTokenIDs.count,
                result: Result {
                    try diagnostics.run(tokenIDs: prefixTokenIDs, topK: topK)
                }
            )
        }
    }
    #endif

    private func makeRuntime() throws -> any InferenceRuntime {
        switch options.runtime {
        case .mock:
            return MockStreamingRuntime(
                tokens: options.mockTokens,
                generatedTokenIDs: options.mockTokenIDs
            )
        case .coreML:
            return try makeCoreMLRuntime()
        }
    }

    private func makeCoreMLRuntime() throws -> any InferenceRuntime {
        #if canImport(CoreML)
        if options.loadOnly {
            return try makeCoreMLLoadProbeRuntime()
        }

        return CoreMLPrefillDecodeRuntime(
            bundle: try makeCoreMLBundle(),
            tokenizer: try makeCoreMLTokenizer()
        )
        #else
        throw RuntimeBenchmarkCommandError.unsupportedRuntime("Core ML is unavailable on this platform")
        #endif
    }

    #if canImport(CoreML)
    private func makeCoreMLBundle() throws -> CoreMLPrefillDecodeBundle {
        let prefillURL = try requiredURL(options.prefillModelURL, "--prefill")
        let decodeURL = try resolvedCoreMLDecodeURL(prefillURL: prefillURL)
        switch options.coreMLGraphInterface {
        case .explicitKV:
            return CoreMLPrefillDecodeBundle.miniCPMExplicitKV(
                prefillModelURL: prefillURL,
                decodeModelURL: decodeURL,
                maxPromptTokens: options.contextVariant
            )
        case .statefulKV:
            return CoreMLPrefillDecodeBundle(
                prefillModelURL: prefillURL,
                decodeModelURL: decodeURL,
                maxPromptTokens: options.contextVariant,
                graphInterface: .statefulKV(layerCount: 24, kvHeads: 2, headDimension: 128),
                decodeTokenInputName: "input_ids",
                decodePositionInputName: "position_ids"
            )
        case .statefulStepKV:
            return CoreMLPrefillDecodeBundle(
                prefillModelURL: prefillURL,
                decodeModelURL: decodeURL,
                maxPromptTokens: options.contextVariant,
                graphInterface: .statefulStepKV(layerCount: 24, kvHeads: 2, headDimension: 128),
                decodeTokenInputName: "input_ids",
                decodePositionInputName: "position_ids"
            )
        }
    }

    private func makeCoreMLTokenizer() throws -> any TextTokenizer {
        try MiniCPMBytePairTokenizer(
            tokenizerJSONURL: requiredURL(options.tokenizerURL, "--tokenizer"),
            addBosToken: true
        )
    }

    private func makeCoreMLLoadProbeRuntime() throws -> any InferenceRuntime {
        switch options.coreMLLoadTarget {
        case .both:
            let prefillURL = try requiredURL(options.prefillModelURL, "--prefill")
            return CoreMLLoadProbeRuntime(modelURLs: [
                prefillURL,
                try resolvedCoreMLDecodeURL(prefillURL: prefillURL)
            ])
        case .prefill:
            return CoreMLLoadProbeRuntime(modelURLs: [
                try requiredURL(options.prefillModelURL, "--prefill")
            ])
        case .decode:
            let prefillURL = try requiredURL(options.prefillModelURL, "--prefill")
            return CoreMLLoadProbeRuntime(modelURLs: [
                try resolvedCoreMLDecodeURL(prefillURL: prefillURL)
            ])
        }
    }
    #endif

    private func resolvedCoreMLDecodeURL(prefillURL: URL) throws -> URL {
        switch options.coreMLGraphInterface {
        case .explicitKV:
            return try requiredURL(options.decodeModelURL, "--decode")
        case .statefulKV, .statefulStepKV:
            let decodeURL = options.decodeModelURL ?? prefillURL
            guard decodeURL.standardizedFileURL == prefillURL.standardizedFileURL else {
                throw RuntimeBenchmarkCommandError.invalidOption(
                    "\(options.coreMLGraphInterface.rawValue) requires --decode to match --prefill when --decode is provided"
                )
            }
            return prefillURL
        }
    }

    private func makeConfiguration() -> RuntimeBenchmarkConfiguration {
        RuntimeBenchmarkConfiguration(
            id: options.configurationID,
            sourceModelId: options.sourceModelID,
            runtime: runtimeIdentifier,
            deviceProfile: options.deviceProfile,
            contextVariant: options.contextVariant,
            artifact: makeArtifact()
        )
    }

    private var runtimeIdentifier: String {
        switch options.runtime {
        case .mock:
            "mock"
        case .coreML:
            "coreml-mlprogram"
        }
    }

    private func makeArtifact() -> RuntimeBenchmarkArtifact? {
        guard options.runtime == .coreML,
              let prefillURL = options.prefillModelURL
        else {
            return nil
        }

        let decodeURL = (try? resolvedCoreMLDecodeURL(prefillURL: prefillURL)) ?? options.decodeModelURL ?? prefillURL
        let tokenizerURL = options.tokenizerURL
        return RuntimeBenchmarkArtifact(
            quantizationPolicyID: options.policyID,
            graphInterface: options.coreMLGraphInterface.rawValue,
            prefillModelPath: displayPath(prefillURL),
            decodeModelPath: displayPath(decodeURL),
            tokenizerPath: tokenizerURL.map(displayPath),
            prefillSizeBytes: try? byteCount(at: prefillURL),
            decodeSizeBytes: try? byteCount(at: decodeURL),
            tokenizerSizeBytes: tokenizerURL.flatMap { try? byteCount(at: $0) }
        )
    }

    private func write(report: RuntimeBenchmarkReport, to outputURL: URL) throws {
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try Self.encode(report: report).write(to: outputURL)
    }

    private func write(diagnosticsReport: CoreMLDiagnosticsReport, to outputURL: URL) throws {
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try Self.encode(diagnosticsReport: diagnosticsReport).write(to: outputURL)
    }

    private func write(sensitivityReport: QuantizationSensitivityReport, to outputURL: URL) throws {
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try Self.encode(sensitivityReport: sensitivityReport).write(to: outputURL)
    }

    private func displayPath(_ url: URL) -> String {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .standardizedFileURL
        let standardized = url.standardizedFileURL
        if let relativePath = standardized.path(percentEncoded: false)
            .relativePath(from: root.path(percentEncoded: false)) {
            return relativePath
        }
        return standardized.path(percentEncoded: false)
    }
}

private struct ParsedBenchmarkArguments {
    var runtime: RuntimeBenchmarkRuntime = .coreML
    var promptsURL: URL
    var teacherReferencesURL: URL?
    var outputURL: URL?
    var promptIDs: [String]?
    var promptLimit: Int?
    var maxNewTokens: Int?
    var requireAllReferences = true
    var deviceProfile: DeviceProfile = .watchSE2
    var contextVariant = 16
    var configurationID = "watchlm-benchmark"
    var sourceModelID = "openbmb/MiniCPM5-1B"
    var policyID = "manual"
    var prefillModelURL: URL?
    var decodeModelURL: URL?
    var tokenizerURL: URL?
    var coreMLGraphInterface: CoreMLBenchmarkGraphInterface = .explicitKV
    var diagnosticsTopK: Int?
    var diagnosticsPrefixLengths: [Int]?
    var sensitivityBaselineURL: URL?
    var sensitivityCandidateURL: URL?
    var loadOnly = false
    var coreMLLoadTarget: CoreMLLoadTarget = .both
    var mockTokens = ["A"]
    var mockTokenIDs: [Int32] = [1]
    private let currentDirectory: URL

    init(currentDirectory: URL) {
        self.currentDirectory = currentDirectory
        promptsURL = currentDirectory.appending(path: "tools/benchmark/fixtures/benchmark-prompts.json")
    }

    func resolve(_ path: String) -> URL {
        let url = URL(fileURLWithPath: NSString(string: path).expandingTildeInPath)
        if url.path(percentEncoded: false).hasPrefix("/") {
            return url
        }
        return currentDirectory.appending(path: path)
    }

    func options() -> RuntimeBenchmarkCommandOptions {
        RuntimeBenchmarkCommandOptions(
            runtime: runtime,
            promptsURL: promptsURL,
            teacherReferencesURL: teacherReferencesURL,
            outputURL: outputURL,
            promptIDs: promptIDs,
            promptLimit: promptLimit,
            maxNewTokens: maxNewTokens,
            requireAllReferences: requireAllReferences,
            deviceProfile: deviceProfile,
            contextVariant: contextVariant,
            configurationID: configurationID,
            sourceModelID: sourceModelID,
            policyID: policyID,
            prefillModelURL: prefillModelURL,
            decodeModelURL: decodeModelURL,
            tokenizerURL: tokenizerURL,
            coreMLGraphInterface: coreMLGraphInterface,
            diagnosticsTopK: diagnosticsTopK,
            diagnosticsPrefixLengths: diagnosticsPrefixLengths,
            sensitivityBaselineURL: sensitivityBaselineURL,
            sensitivityCandidateURL: sensitivityCandidateURL,
            loadOnly: loadOnly,
            coreMLLoadTarget: coreMLLoadTarget,
            mockTokens: mockTokens,
            mockTokenIDs: mockTokenIDs
        )
    }
}

private func value(after option: String, in arguments: [String], at index: inout Int) throws -> String {
    let valueIndex = index + 1
    guard arguments.indices.contains(valueIndex), !arguments[valueIndex].hasPrefix("--") else {
        throw RuntimeBenchmarkCommandError.missingOption(option)
    }
    index = valueIndex
    return arguments[valueIndex]
}

private func parsePositiveInt(_ value: String, option: String) throws -> Int {
    guard let parsed = Int(value), parsed > 0 else {
        throw RuntimeBenchmarkCommandError.invalidOption("\(option) must be a positive integer")
    }
    return parsed
}

private func parseStringList(_ value: String, option: String) throws -> [String] {
    let values = value.split(separator: ",").map(String.init).filter { !$0.isEmpty }
    guard !values.isEmpty else {
        throw RuntimeBenchmarkCommandError.invalidOption("\(option) must contain at least one value")
    }
    return values
}

private func parsePositiveIntList(_ value: String, option: String) throws -> [Int] {
    let values = try value.split(separator: ",").map { token throws -> Int in
        let trimmed = token.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let parsed = Int(trimmed), parsed > 0 else {
            throw RuntimeBenchmarkCommandError.invalidOption("\(option) must contain comma-separated positive integers")
        }
        return parsed
    }
    guard !values.isEmpty else {
        throw RuntimeBenchmarkCommandError.invalidOption("\(option) must contain at least one value")
    }
    return values
}

private func parseInt32List(_ value: String, option: String) throws -> [Int32] {
    let values = try value.split(separator: ",").map { token throws -> Int32 in
        guard let parsed = Int32(token.trimmingCharacters(in: .whitespacesAndNewlines)) else {
            throw RuntimeBenchmarkCommandError.invalidOption("\(option) must contain comma-separated integer ids")
        }
        return parsed
    }
    guard !values.isEmpty else {
        throw RuntimeBenchmarkCommandError.invalidOption("\(option) must contain at least one id")
    }
    return values
}

private func requiredURL(_ url: URL?, _ option: String) throws -> URL {
    guard let url else {
        throw RuntimeBenchmarkCommandError.missingOption(option)
    }
    return url
}

private func byteCount(at url: URL) throws -> Int64 {
    let values = try url.resourceValues(forKeys: [.isDirectoryKey, .isRegularFileKey, .fileSizeKey])
    if values.isDirectory == true {
        guard let enumerator = FileManager.default.enumerator(
            at: url,
            includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else {
            return 0
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
    return Int64(values.fileSize ?? 0)
}

#if canImport(CoreML)
private final class CoreMLLoadProbeRuntime: InferenceRuntime, @unchecked Sendable {
    private let modelURLs: [URL]
    private var loadedModels: [MLModel] = []

    init(modelURLs: [URL]) {
        self.modelURLs = modelURLs
    }

    func load() async throws -> RuntimeTiming {
        let started = Date()
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        loadedModels = try modelURLs.map { url in
            try loadBenchmarkModel(at: url, configuration: configuration)
        }
        return RuntimeTiming(loadMs: elapsedMilliseconds(since: started))
    }

    func generate(
        request: InferenceRequest,
        shouldCancel: @Sendable () -> Bool
    ) async throws -> InferenceResult {
        throw InferenceRuntimeError.invalidInput(message: "Core ML load probe does not generate tokens.")
    }
}

private func loadBenchmarkModel(at url: URL, configuration: MLModelConfiguration) throws -> MLModel {
    #if os(macOS)
    if url.pathExtension == "mlpackage" || url.pathExtension == "mlmodel" {
        let compiledURL = try MLModel.compileModel(at: url)
        return try MLModel(contentsOf: compiledURL, configuration: configuration)
    }
    #endif

    return try MLModel(contentsOf: url, configuration: configuration)
}

private func elapsedMilliseconds(since started: Date) -> Double {
    let elapsed = Date().timeIntervalSince(started) * 1000
    return (elapsed * 1000).rounded() / 1000
}
#endif

private extension Optional {
    func orThrowInvalid(_ message: String) throws -> Wrapped {
        guard let value = self else {
            throw RuntimeBenchmarkCommandError.invalidOption(message)
        }
        return value
    }
}

private extension String {
    func relativePath(from root: String) -> String? {
        guard hasPrefix(root) else {
            return nil
        }
        let start = index(startIndex, offsetBy: root.count)
        let suffix = self[start...]
        if suffix.isEmpty {
            return "."
        }
        return suffix.first == "/" ? String(suffix.dropFirst()) : String(suffix)
    }
}
