import Foundation
import Testing
@testable import WatchLMBenchmarkSupport
@testable import WatchLMCore

@Test func runtimeBenchmarkCommandMergesTeacherSidecarAndWritesMockReport() async throws {
    let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let promptsURL = root.appending(path: "tools/benchmark/fixtures/benchmark-prompts.json")
    let promptSuite = try RuntimeBenchmarkPromptSuite.load(from: promptsURL)
    let selectedPrompts = Array(promptSuite.prompts.prefix(2))
    let temporaryDirectory = try makeTemporaryDirectory()
    let referencesURL = temporaryDirectory.appending(path: "teacher-references.json")
    let outputURL = temporaryDirectory.appending(path: "benchmark-report.json")

    let references = RuntimeBenchmarkQualityReferenceSuite(
        schemaVersion: 1,
        source: "mock-teacher",
        references: selectedPrompts.map { prompt in
            RuntimeBenchmarkPromptQualityReference(promptID: prompt.id, tokenIDs: [10, 11, 12])
        } + [
            RuntimeBenchmarkPromptQualityReference(promptID: "en-short-001", tokenIDs: [12, 13])
        ]
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(references).write(to: referencesURL)

    let report = try await RuntimeBenchmarkCommand(
        options: RuntimeBenchmarkCommandOptions(
            runtime: .mock,
            promptsURL: promptsURL,
            teacherReferencesURL: referencesURL,
            outputURL: outputURL,
            promptLimit: 2,
            maxNewTokens: 2,
            requireAllReferences: true,
            deviceProfile: .watchSE2,
            contextVariant: 16,
            configurationID: "mock-sidecar-cli",
            sourceModelID: "openbmb/MiniCPM5-1B",
            policyID: "mock-policy",
            mockTokens: ["A", "B"],
            mockTokenIDs: [10, 11]
        )
    ).run()

    #expect(report.configuration.id == "mock-sidecar-cli")
    #expect(report.summary.succeededPromptCount == 2)
    #expect(report.summary.averageTokenAgreement == 1.0)
    #expect(FileManager.default.fileExists(atPath: outputURL.path))

    let writtenReport = try JSONDecoder().decode(RuntimeBenchmarkReport.self, from: Data(contentsOf: outputURL))
    #expect(writtenReport == report)
}

@Test func runtimeBenchmarkCommandCanSelectPromptIDsInRequestedOrder() async throws {
    let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let promptsURL = root.appending(path: "tools/benchmark/fixtures/benchmark-prompts.json")
    let temporaryDirectory = try makeTemporaryDirectory()
    let referencesURL = temporaryDirectory.appending(path: "teacher-references.json")
    let outputURL = temporaryDirectory.appending(path: "benchmark-report.json")

    let references = RuntimeBenchmarkQualityReferenceSuite(
        schemaVersion: 1,
        source: "mock-teacher",
        references: [
            RuntimeBenchmarkPromptQualityReference(promptID: "zh-short-001", tokenIDs: [10, 11]),
            RuntimeBenchmarkPromptQualityReference(promptID: "watch-utility-001", tokenIDs: [10, 11])
        ]
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(references).write(to: referencesURL)

    let options = try RuntimeBenchmarkCommandOptions.parse(
        [
            "--runtime", "mock",
            "--prompts", promptsURL.path,
            "--teacher", referencesURL.path,
            "--output", outputURL.path,
            "--prompt-ids", "watch-utility-001,zh-short-001",
            "--max-new-tokens", "2",
            "--mock-tokens", "A,B",
            "--mock-token-ids", "10,11"
        ]
    )
    let report = try await RuntimeBenchmarkCommand(options: options).run()

    #expect(report.promptResults.map(\.promptID) == ["watch-utility-001", "zh-short-001"])
    #expect(report.summary.averageTokenAgreement == 1.0)
}

@Test func runtimeBenchmarkCommandCanRunLoadOnlyBenchmarkWithoutPrompts() async throws {
    let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let promptsURL = root.appending(path: "tools/benchmark/fixtures/benchmark-prompts.json")
    let temporaryDirectory = try makeTemporaryDirectory()
    let outputURL = temporaryDirectory.appending(path: "load-only-report.json")

    let options = try RuntimeBenchmarkCommandOptions.parse(
        [
            "--runtime", "mock",
            "--prompts", promptsURL.path,
            "--output", outputURL.path,
            "--load-only",
            "--coreml-load-target", "prefill"
        ]
    )
    let report = try await RuntimeBenchmarkCommand(options: options).run()

    #expect(options.loadOnly)
    #expect(options.coreMLLoadTarget == .prefill)
    #expect(report.promptResults.isEmpty)
    #expect(report.summary.promptCount == 0)
    #expect(FileManager.default.fileExists(atPath: outputURL.path))
}

@Test func runtimeBenchmarkCommandParsesStatefulCoreMLGraphInterface() throws {
    let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let modelURL = root.appending(path: "Models/MiniCPM5/stateful-256.mlmodelc")
    let tokenizerURL = root.appending(path: "Models/MiniCPM5/tokenizer.json")

    let options = try RuntimeBenchmarkCommandOptions.parse(
        [
            "--runtime", "coreml",
            "--prefill", modelURL.path,
            "--tokenizer", tokenizerURL.path,
            "--coreml-graph-interface", "stateful-kv",
            "--context", "256"
        ],
        currentDirectory: root
    )

    #expect(options.coreMLGraphInterface == .statefulKV)
    #expect(options.prefillModelURL == modelURL)
    #expect(options.decodeModelURL == nil)
    #expect(options.contextVariant == 256)
}

@Test func runtimeBenchmarkCommandParsesStatefulStepCoreMLGraphInterface() throws {
    let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let modelURL = root.appending(path: "Models/MiniCPM5/stateful-step-256.mlmodelc")
    let tokenizerURL = root.appending(path: "Models/MiniCPM5/tokenizer.json")

    let options = try RuntimeBenchmarkCommandOptions.parse(
        [
            "--runtime", "coreml",
            "--prefill", modelURL.path,
            "--tokenizer", tokenizerURL.path,
            "--coreml-graph-interface", "stateful-step-kv",
            "--context", "256"
        ],
        currentDirectory: root
    )

    #expect(options.coreMLGraphInterface == .statefulStepKV)
    #expect(options.prefillModelURL == modelURL)
    #expect(options.decodeModelURL == nil)
}
