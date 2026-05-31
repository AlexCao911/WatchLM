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

@Test func runtimeBenchmarkCommandCanRunCalibrationPromptSuite() async throws {
    let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let calibrationPromptsURL = root.appending(path: "tools/benchmark/fixtures/calibration-prompts.json")
    let temporaryDirectory = try makeTemporaryDirectory()
    let outputURL = temporaryDirectory.appending(path: "calibration-benchmark-report.json")

    let options = try RuntimeBenchmarkCommandOptions.parse(
        [
            "--runtime", "mock",
            "--calibration-prompts", calibrationPromptsURL.path,
            "--output", outputURL.path,
            "--prompt-ids", "cal-watch-utility-001,cal-zh-short-001",
            "--max-new-tokens", "2",
            "--mock-tokens", "A,B",
            "--mock-token-ids", "10,11"
        ],
        currentDirectory: root
    )
    let report = try await RuntimeBenchmarkCommand(options: options).run()

    #expect(options.calibrationPromptsURL == calibrationPromptsURL)
    #expect(report.promptResults.map(\.promptID) == ["cal-watch-utility-001", "cal-zh-short-001"])
    #expect(report.promptResults.allSatisfy { $0.generatedTokenIDs == [10, 11] })
    #expect(report.summary.promptCount == 2)
    #expect(FileManager.default.fileExists(atPath: outputURL.path))
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

@Test func runtimeBenchmarkCommandParsesRuntimeCandidateGraphDimensionsAndSpecialTokens() throws {
    let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let modelURL = root.appending(path: "Models/Qwen3/stateful-step-256.mlmodelc")
    let tokenizerURL = root.appending(path: "Models/Qwen3/tokenizer.json")

    let options = try RuntimeBenchmarkCommandOptions.parse(
        [
            "--runtime", "coreml",
            "--prefill", modelURL.path,
            "--tokenizer", tokenizerURL.path,
            "--coreml-graph-interface", "stateful-step-kv",
            "--coreml-layer-count", "28",
            "--coreml-kv-heads", "8",
            "--coreml-head-dim", "128",
            "--coreml-compute-units", "cpu-only",
            "--tokenizer-add-bos", "false",
            "--tokenizer-bos-token-id", "151643",
            "--tokenizer-eos-token-ids", "151645",
            "--chat-template", "qwen3-nonthinking",
            "--context", "256"
        ],
        currentDirectory: root
    )

    #expect(options.coreMLLayerCount == 28)
    #expect(options.coreMLKVHeads == 8)
    #expect(options.coreMLHeadDimension == 128)
    #expect(options.coreMLComputeUnits == .cpuOnly)
    #expect(options.tokenizerAddBOS == false)
    #expect(options.tokenizerBOSTokenID == 151643)
    #expect(options.tokenizerEOSTokenIDs == [151645])
    #expect(options.chatTemplate == .qwen3NonThinking)
}

@Test func runtimeBenchmarkCommandCanResolveQwenCoreMLOptionsFromManifest() throws {
    let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let manifestURL = root.appending(path: "tools/validation/fixtures/qwen3-0.6b-explicit-kv-model-manifest.json")
    let assetBaseURL = root.appending(path: "artifacts/runtime-candidates")

    let options = try RuntimeBenchmarkCommandOptions.parse(
        [
            "--manifest", manifestURL.path,
            "--asset-base", assetBaseURL.path,
            "--device-profile", "watch-se-2",
            "--coreml-compute-units", "cpu-only"
        ],
        currentDirectory: root
    )

    #expect(options.runtime == .coreML)
    #expect(options.manifestURL == manifestURL)
    #expect(options.assetBaseURL?.standardizedFileURL.path() == assetBaseURL.standardizedFileURL.path())
    #expect(options.sourceModelID == "Qwen/Qwen3-0.6B")
    #expect(options.contextVariant == 256)
    #expect(options.prefillModelURL == assetBaseURL.appending(path: "Models/Qwen3/prefill-kv-256-int8.mlpackage"))
    #expect(options.decodeModelURL == assetBaseURL.appending(path: "Models/Qwen3/decode-256-int8.mlpackage"))
    #expect(options.tokenizerURL == assetBaseURL.appending(path: "Models/Qwen3/tokenizer.json"))
    #expect(options.coreMLGraphInterface == .explicitKV)
    #expect(options.coreMLLayerCount == 28)
    #expect(options.coreMLKVHeads == 8)
    #expect(options.coreMLHeadDimension == 128)
    #expect(options.coreMLComputeUnits == .cpuOnly)
    #expect(options.tokenizerAddBOS == false)
    #expect(options.tokenizerBOSTokenID == 151643)
    #expect(options.tokenizerEOSTokenIDs == [151645])
    #expect(options.chatTemplate == .qwen3NonThinking)
}

@Test func runtimeBenchmarkCommandCanResolveQwenStatefulCoreMLOptionsFromManifest() throws {
    let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let manifestURL = root.appending(path: "tools/validation/fixtures/qwen3-0.6b-stateful-step-model-manifest.json")
    let assetBaseURL = root.appending(path: "artifacts/runtime-candidates")

    let options = try RuntimeBenchmarkCommandOptions.parse(
        [
            "--manifest", manifestURL.path,
            "--asset-base", assetBaseURL.path,
            "--device-profile", "watch-se-2",
            "--coreml-compute-units", "cpu-only"
        ],
        currentDirectory: root
    )

    #expect(options.runtime == .coreML)
    #expect(options.assetBaseURL?.standardizedFileURL.path() == assetBaseURL.standardizedFileURL.path())
    #expect(options.sourceModelID == "Qwen/Qwen3-0.6B")
    #expect(options.contextVariant == 256)
    #expect(options.prefillModelURL == assetBaseURL.appending(path: "Models/Qwen3/stateful-step-kv-256-fp32-compute-int8.mlpackage"))
    #expect(options.decodeModelURL == options.prefillModelURL)
    #expect(options.tokenizerURL == assetBaseURL.appending(path: "Models/Qwen3/tokenizer.json"))
    #expect(options.coreMLGraphInterface == .statefulStepKV)
    #expect(options.coreMLGraphSchema?.interface == "stateful-step-kv")
    #expect(options.coreMLGraphSchema?.decode.tokenID == "input_ids")
    #expect(options.coreMLGraphSchema?.decode.positionID == "position_ids")
    #expect(options.coreMLGraphSchema?.layerCount == 28)
    #expect(options.coreMLGraphSchema?.kvHeads == 8)
    #expect(options.coreMLGraphSchema?.headDimension == 128)
    #expect(options.coreMLKVCacheUpdateStrategy == .slotRing)
    #expect(options.tokenizerAddBOS == false)
    #expect(options.tokenizerBOSTokenID == 151643)
    #expect(options.tokenizerEOSTokenIDs == [151645])
    #expect(options.chatTemplate == .qwen3NonThinking)
}

@Test func runtimeBenchmarkCommandCanWriteQwenStatefulDeviceStagingPlan() throws {
    let root = try makeTemporaryDirectory()
    let assetBaseURL = root.appending(path: "runtime-candidates", directoryHint: .isDirectory)
    let modelDirectory = assetBaseURL
        .appending(path: "Models", directoryHint: .isDirectory)
        .appending(path: "Qwen3", directoryHint: .isDirectory)
    let statefulURL = modelDirectory
        .appending(path: "stateful-step-kv-256-fp32-compute-int8.mlpackage", directoryHint: .isDirectory)
    let tokenizerURL = modelDirectory.appending(path: "tokenizer.json")
    try FileManager.default.createDirectory(at: statefulURL, withIntermediateDirectories: true)
    try Data("qwen-stateful".utf8).write(to: statefulURL.appending(path: "Manifest.json"))
    try minimalTokenizerJSONData().write(to: tokenizerURL)

    var manifest = makeQwenStatefulTestManifest()
    manifest.asset.prefillSHA256 = try ArtifactDigest.sha256Hex(for: statefulURL)
    manifest.asset.decodeSHA256 = try ArtifactDigest.sha256Hex(for: statefulURL)
    manifest.asset.tokenizerSHA256 = try ArtifactDigest.sha256Hex(for: tokenizerURL)
    let manifestURL = root.appending(path: "qwen-manifest.json")
    let outputURL = root.appending(path: "qwen-staging-plan.json")
    try JSONEncoder().encode(manifest).write(to: manifestURL)

    let options = try RuntimeBenchmarkCommandOptions.parse(
        [
            "--manifest", manifestURL.path,
            "--asset-base", assetBaseURL.path,
            "--device-profile", "watch-se-2",
            "--staging-plan",
            "--output", outputURL.path
        ],
        currentDirectory: root
    )
    let plan = try RuntimeBenchmarkCommand(options: options).runStagingPlan()

    #expect(options.stagingPlanOnly)
    #expect(plan.deviceProfile == .watchSE2)
    #expect(plan.items.map(\.destinationRelativePath) == [
        "model-manifest.json",
        "Models/Qwen3/stateful-step-kv-256-fp32-compute-int8.mlpackage",
        "Models/Qwen3/tokenizer.json"
    ])
    #expect(plan.items[1].purposes == [.prefill, .decode])
    #expect(plan.items[1].actualSHA256 == manifest.asset.prefillSHA256)
    #expect(plan.items[2].actualSHA256 == manifest.asset.tokenizerSHA256)
    #expect(FileManager.default.fileExists(atPath: outputURL.path))
}

@Test func runtimeBenchmarkCommandParsesCoreMLDiagnosticsTopK() throws {
    let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let modelURL = root.appending(path: "Models/MiniCPM5/stateful-step-256.mlmodelc")
    let tokenizerURL = root.appending(path: "Models/MiniCPM5/tokenizer.json")

    let options = try RuntimeBenchmarkCommandOptions.parse(
        [
            "--runtime", "coreml",
            "--prefill", modelURL.path,
            "--tokenizer", tokenizerURL.path,
            "--coreml-graph-interface", "stateful-step-kv",
            "--diagnostics-top-k", "5",
            "--prompt-ids", "en-short-001"
        ],
        currentDirectory: root
    )

    #expect(options.diagnosticsTopK == 5)
    #expect(options.coreMLGraphInterface == .statefulStepKV)
    #expect(options.promptIDs == ["en-short-001"])
}

@Test func runtimeBenchmarkCommandParsesCoreMLDiagnosticsPrefixLengths() throws {
    let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let modelURL = root.appending(path: "Models/MiniCPM5/stateful-step-256.mlmodelc")
    let tokenizerURL = root.appending(path: "Models/MiniCPM5/tokenizer.json")

    let options = try RuntimeBenchmarkCommandOptions.parse(
        [
            "--runtime", "coreml",
            "--prefill", modelURL.path,
            "--tokenizer", tokenizerURL.path,
            "--coreml-graph-interface", "stateful-step-kv",
            "--diagnostics-top-k", "5",
            "--diagnostics-prefix-lengths", "1,2,4"
        ],
        currentDirectory: root
    )

    #expect(options.diagnosticsPrefixLengths == [1, 2, 4])
}

@Test func runtimeBenchmarkCommandParsesSensitivityComparisonInputs() throws {
    let temporaryDirectory = try makeTemporaryDirectory()
    let baselineURL = temporaryDirectory.appending(path: "fp16-diagnostics.json")
    let candidateURL = temporaryDirectory.appending(path: "candidate-diagnostics.json")
    let outputURL = temporaryDirectory.appending(path: "sensitivity-report.json")

    let options = try RuntimeBenchmarkCommandOptions.parse(
        [
            "--sensitivity-baseline", baselineURL.path,
            "--sensitivity-candidate", candidateURL.path,
            "--output", outputURL.path
        ]
    )

    #expect(options.sensitivityBaselineURL == baselineURL)
    #expect(options.sensitivityCandidateURL == candidateURL)
    #expect(options.outputURL == outputURL)
    #expect(options.runsSensitivityComparison)
}

@Test func runtimeBenchmarkCommandComparesDiagnosticsReports() throws {
    let temporaryDirectory = try makeTemporaryDirectory()
    let baselineURL = temporaryDirectory.appending(path: "fp16-diagnostics.json")
    let candidateURL = temporaryDirectory.appending(path: "candidate-diagnostics.json")
    let outputURL = temporaryDirectory.appending(path: "sensitivity-report.json")

    try writeDiagnosticsReport(
        policyID: "stateful-step-kv-256-fp16",
        promptResults: [
            CoreMLDiagnosticPromptResult(
                promptID: "en-short-001",
                category: "en_short_instruction",
                language: "en",
                requestedPrefixTokenCount: 1,
                prefixTokenCount: 1,
                prefillTopK: commandTopK([5, 24, 49, 11127, 45050])
            ),
            CoreMLDiagnosticPromptResult(
                promptID: "en-short-001",
                category: "en_short_instruction",
                language: "en",
                requestedPrefixTokenCount: 2,
                prefixTokenCount: 2,
                prefillTopK: commandTopK([285, 1070, 316, 3212, 976])
            )
        ],
        to: baselineURL
    )
    try writeDiagnosticsReport(
        policyID: "stateful-step-layer8-15-v-layer11-12-qk-int4",
        promptResults: [
            CoreMLDiagnosticPromptResult(
                promptID: "en-short-001",
                category: "en_short_instruction",
                language: "en",
                requestedPrefixTokenCount: 1,
                prefixTokenCount: 1,
                prefillTopK: commandTopK([5, 24, 49, 11127, 45050])
            ),
            CoreMLDiagnosticPromptResult(
                promptID: "en-short-001",
                category: "en_short_instruction",
                language: "en",
                requestedPrefixTokenCount: 2,
                prefixTokenCount: 2,
                prefillTopK: commandTopK([5, 24, 5298, 1207, 20773])
            )
        ],
        to: candidateURL
    )

    let report = try RuntimeBenchmarkCommand(
        options: RuntimeBenchmarkCommandOptions(
            promptsURL: temporaryDirectory.appending(path: "unused-prompts.json"),
            outputURL: outputURL,
            sensitivityBaselineURL: baselineURL,
            sensitivityCandidateURL: candidateURL
        )
    ).runSensitivityComparison()

    #expect(report.baselinePolicyID == "stateful-step-kv-256-fp16")
    #expect(report.candidatePolicyID == "stateful-step-layer8-15-v-layer11-12-qk-int4")
    #expect(report.summary.averagePrefillTopKOverlapRatio == 0.5)
    #expect(report.summary.firstZeroPrefillOverlapPrefixTokenCount == 2)
    #expect(!report.gate.ok)
    #expect(FileManager.default.fileExists(atPath: outputURL.path))

    let writtenReport = try JSONDecoder().decode(
        QuantizationSensitivityReport.self,
        from: Data(contentsOf: outputURL)
    )
    #expect(writtenReport == report)
}

private func writeDiagnosticsReport(
    policyID: String,
    promptResults: [CoreMLDiagnosticPromptResult],
    to outputURL: URL
) throws {
    let report = CoreMLDiagnosticsReport(
        configuration: RuntimeBenchmarkConfiguration(
            id: "\(policyID)-diagnostics",
            sourceModelId: "openbmb/MiniCPM5-1B",
            runtime: "coreml-mlprogram",
            deviceProfile: .watchSE2,
            contextVariant: 256,
            artifact: RuntimeBenchmarkArtifact(
                quantizationPolicyID: policyID,
                graphInterface: "stateful-step-kv",
                prefillModelPath: "\(policyID).mlpackage",
                decodeModelPath: "\(policyID).mlpackage"
            )
        ),
        topK: 5,
        promptResults: promptResults
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(report).write(to: outputURL)
}

private func commandTopK(_ tokenIDs: [Int32]) -> [TokenLogit] {
    tokenIDs.enumerated().map { index, tokenID in
        TokenLogit(tokenID: tokenID, logit: Double(tokenIDs.count - index))
    }
}
