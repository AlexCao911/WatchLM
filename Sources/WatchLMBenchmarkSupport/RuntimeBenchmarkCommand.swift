import Foundation
import WatchLMCore

public enum RuntimeBenchmarkRuntime: String, Codable, Equatable, Sendable {
    case mock
    case coreML = "coreml"
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

public struct RuntimeBenchmarkCommandOptions: Equatable, Sendable {
    public var runtime: RuntimeBenchmarkRuntime
    public var promptsURL: URL
    public var teacherReferencesURL: URL?
    public var outputURL: URL?
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
    public var mockTokens: [String]
    public var mockTokenIDs: [Int32]

    public init(
        runtime: RuntimeBenchmarkRuntime = .coreML,
        promptsURL: URL,
        teacherReferencesURL: URL? = nil,
        outputURL: URL? = nil,
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
        mockTokens: [String] = ["A"],
        mockTokenIDs: [Int32] = [1]
    ) {
        self.runtime = runtime
        self.promptsURL = promptsURL
        self.teacherReferencesURL = teacherReferencesURL
        self.outputURL = outputURL
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
        self.mockTokens = mockTokens
        self.mockTokenIDs = mockTokenIDs
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

    Options:
      --prompts PATH                 Prompt suite JSON. Defaults to tools/benchmark/fixtures/benchmark-prompts.json.
      --teacher PATH                 Teacher reference sidecar JSON.
      --output PATH                  Write RuntimeBenchmarkReport JSON to this path. Without it, JSON is printed to stdout.
      --prompt-limit N               Run only the first N prompts.
      --max-new-tokens N             Cap each prompt's maxNewTokens for smoke runs.
      --allow-missing-references     Do not require every selected prompt to have teacher tokens.
      --device-profile watch-se-2|watch-se-3
      --context N
      --policy-id ID
      --id ID
    """

    private let options: RuntimeBenchmarkCommandOptions

    public init(options: RuntimeBenchmarkCommandOptions) {
        self.options = options
    }

    public func run() async throws -> RuntimeBenchmarkReport {
        let prompts = try loadPrompts()
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

    public static func encode(report: RuntimeBenchmarkReport) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(report)
    }

    private func loadPrompts() throws -> [RuntimeBenchmarkPrompt] {
        var suite = try RuntimeBenchmarkPromptSuite.load(from: options.promptsURL)
        _ = try suite.validatedPrompts()

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
        let prefillURL = try requiredURL(options.prefillModelURL, "--prefill")
        let decodeURL = try requiredURL(options.decodeModelURL, "--decode")
        let tokenizerURL = try requiredURL(options.tokenizerURL, "--tokenizer")
        let bundle = CoreMLPrefillDecodeBundle.miniCPMExplicitKV(
            prefillModelURL: prefillURL,
            decodeModelURL: decodeURL,
            maxPromptTokens: options.contextVariant
        )
        return CoreMLPrefillDecodeRuntime(
            bundle: bundle,
            tokenizer: try MiniCPMBytePairTokenizer(tokenizerJSONURL: tokenizerURL, addBosToken: true)
        )
        #else
        throw RuntimeBenchmarkCommandError.unsupportedRuntime("Core ML is unavailable on this platform")
        #endif
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
              let prefillURL = options.prefillModelURL,
              let decodeURL = options.decodeModelURL
        else {
            return nil
        }

        let tokenizerURL = options.tokenizerURL
        return RuntimeBenchmarkArtifact(
            quantizationPolicyID: options.policyID,
            graphInterface: "logits-layered-kv",
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
