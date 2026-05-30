import Foundation
import Testing
@testable import WatchLMCore

#if canImport(CoreML)
import CoreML

@Test func coreMLPrefillDecodeRuntimeRunsSplitPredictionsWithExplicitKVCache() async throws {
    let bundle = CoreMLPrefillDecodeBundle(
        prefillModelURL: try #require(smokeModelURL(named: "SmokePrefill")),
        decodeModelURL: try #require(smokeModelURL(named: "SmokeDecode")),
        maxPromptTokens: 4
    )
    let runtime = CoreMLPrefillDecodeRuntime(
        bundle: bundle,
        tokenizer: FixtureTokenIDTokenizer()
    )

    let loadTiming = try await runtime.load()
    let result = try await runtime.generate(
        request: InferenceRequest(prompt: "A B", maxNewTokens: 3),
        shouldCancel: { false }
    )

    #expect(loadTiming.loadMs >= 0)
    #expect(result.tokens == ["D", "E", "F"])
    #expect(result.generatedTokenIDs == [5, 6, 7])
    #expect(result.terminationReason == .maxTokens)
    #expect(result.text == "DEF")
    #expect(result.timing.prefillMs >= 0)
    #expect(result.timing.decodeStepMs.count == 2)
    print("WATCHLM_PREFILL_DECODE_SMOKE output=\(result.text) load_ms=\(String(format: "%.3f", loadTiming.loadMs)) prefill_ms=\(String(format: "%.3f", result.timing.prefillMs)) decode_tps=\(String(format: "%.2f", result.timing.decodeTokensPerSecond))")
}

@Test func coreMLPrefillDecodeRuntimeRunsLogitsAndLayeredKVGraph() async throws {
    let prefillURL = try #require(smokeModelURL(named: "SmokeLayeredPrefill"))
    let decodeURL = try #require(smokeModelURL(named: "SmokeLayeredDecode"))
    let bundle = CoreMLPrefillDecodeBundle(
        prefillModelURL: prefillURL,
        decodeModelURL: decodeURL,
        maxPromptTokens: 4,
        graphInterface: .logitsAndLayeredKV(layerCount: 1, kvHeads: 1, headDimension: 1),
        decodeTokenInputName: "token_id"
    )
    let runtime = CoreMLPrefillDecodeRuntime(
        bundle: bundle,
        tokenizer: FixtureTokenIDTokenizer()
    )

    let result = try await runtime.generate(
        request: InferenceRequest(prompt: "A B", maxNewTokens: 3),
        shouldCancel: { false }
    )

    #expect(result.tokens == ["D", "E", "F"])
    #expect(result.generatedTokenIDs == [5, 6, 7])
    #expect(result.terminationReason == .maxTokens)
    #expect(result.timing.decodeStepMs.count == 2)
    #expect(result.metrics.prefillSamplingMs >= 0)
    #expect(result.metrics.decodeSamplingStepMs.count == 2)
    #expect(result.metrics.kvAppendStepMs.count == 2)
    #expect(result.metrics.kvCacheUpdateStrategy == .slotRing)
    #expect(result.metrics.kvAppendWriteIndices == [1, 0])
    #expect(result.metrics.kvAppendMovedTokenSlots == [0, 0])
    #expect(result.metrics.kvAppendMovedScalarCounts == [0, 0])
    #expect(result.metrics.totalKVAppendMovedScalarCount == 0)
}

@Test func coreMLPrefillDecodeDiagnosticsExposePrefillAndDecodeTopK() throws {
    let prefillURL = try #require(smokeModelURL(named: "SmokeLayeredPrefill"))
    let decodeURL = try #require(smokeModelURL(named: "SmokeLayeredDecode"))
    let bundle = CoreMLPrefillDecodeBundle(
        prefillModelURL: prefillURL,
        decodeModelURL: decodeURL,
        maxPromptTokens: 4,
        graphInterface: .logitsAndLayeredKV(layerCount: 1, kvHeads: 1, headDimension: 1),
        decodeTokenInputName: "token_id"
    )

    let report = try CoreMLPrefillDecodeDiagnostics(
        bundle: bundle,
        tokenizer: FixtureTokenIDTokenizer()
    ).run(prompt: "A B", topK: 3)

    #expect(report.prefillTokenID == 5)
    #expect(report.firstDecodeTokenID == 6)
    #expect(report.prefillTopK.count == 3)
    #expect(report.decodeTopK.count == 3)
    #expect(report.promptTokenIDs == [2, 3])
}

#if os(macOS)
@Test func coreMLPrefillDecodeRuntimeCanRunLocalRealMiniCPMInt8Artifacts() async throws {
    guard ProcessInfo.processInfo.environment["WATCHLM_RUN_REAL_COREML_TESTS"] == "1" else {
        return
    }

    let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let prefillURL = root.appending(path: "artifacts/coreml/real-minicpm5-prefill-kv-16-int8/prefill-kv-16-int8.mlpackage")
    let decodeURL = root.appending(path: "artifacts/coreml/real-minicpm5-decode-16-int8/decode-16-int8.mlpackage")
    let tokenizerURL = root.appending(path: "artifacts/hf/MiniCPM5-1B/tokenizer.json")
    guard FileManager.default.fileExists(atPath: prefillURL.path),
          FileManager.default.fileExists(atPath: decodeURL.path),
          FileManager.default.fileExists(atPath: tokenizerURL.path)
    else {
        return
    }

    let bundle = CoreMLPrefillDecodeBundle.miniCPMExplicitKV(
        prefillModelURL: prefillURL,
        decodeModelURL: decodeURL,
        maxPromptTokens: 16
    )
    let runtime = CoreMLPrefillDecodeRuntime(
        bundle: bundle,
        tokenizer: try MiniCPMBytePairTokenizer(tokenizerJSONURL: tokenizerURL, addBosToken: true)
    )
    let artifact = RuntimeBenchmarkArtifact(
        quantizationPolicyID: "global-int8",
        graphInterface: "logits-layered-kv",
        prefillModelPath: "artifacts/coreml/real-minicpm5-prefill-kv-16-int8/prefill-kv-16-int8.mlpackage",
        decodeModelPath: "artifacts/coreml/real-minicpm5-decode-16-int8/decode-16-int8.mlpackage",
        tokenizerPath: "artifacts/hf/MiniCPM5-1B/tokenizer.json"
    )
    let report = try await RuntimeBenchmarkRunner().run(
        runtime: runtime,
        configuration: RuntimeBenchmarkConfiguration(
            id: "real-minicpm5-context16-int8-local",
            sourceModelId: "openbmb/MiniCPM5-1B",
            runtime: "coreml-mlprogram",
            deviceProfile: .watchSE2,
            contextVariant: 16,
            artifact: artifact
        ),
        prompts: [
            RuntimeBenchmarkPrompt(
                id: "apple-watch-local-inference-test",
                category: "watch_utility",
                language: "en",
                input: "Apple Watch local inference test.",
                maxNewTokens: 2,
                qualityReference: RuntimeQualityReference(
                    source: "pytorch-teacher-context16-int8-validation",
                    tokenIDs: [242, 38]
                )
            )
        ]
    )
    let result = try #require(report.promptResults.first)

    #expect(report.configuration.artifact == artifact)
    #expect(report.loadTiming.loadMs >= 0)
    #expect(report.summary.succeededPromptCount == 1)
    #expect(report.summary.averageTokenAgreement == 1.0)
    #expect(result.generatedTokenIDs == [242, 38])
    #expect(result.quality?.tokenAgreement == 1.0)
    #expect(result.timing.prefillMs >= 0)
    #expect(result.timing.decodeStepMs.count == 1)
    #expect(result.metrics.prefillSamplingMs >= 0)
    #expect(result.metrics.decodeSamplingStepMs.count == 1)
    #expect(result.metrics.kvAppendStepMs.count == 1)
    #expect(result.metrics.kvAppendWriteIndices.count == 1)
    #expect(result.metrics.kvCacheUpdateStrategy == .slotRing)
}

@Test func coreMLPrefillDecodeDiagnosticsCanRunLocalRealMiniCPMInt8Artifacts() throws {
    guard ProcessInfo.processInfo.environment["WATCHLM_RUN_REAL_COREML_TESTS"] == "1" else {
        return
    }

    let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let prefillURL = root.appending(path: "artifacts/coreml/real-minicpm5-prefill-kv-16-int8/prefill-kv-16-int8.mlpackage")
    let decodeURL = root.appending(path: "artifacts/coreml/real-minicpm5-decode-16-int8/decode-16-int8.mlpackage")
    let tokenizerURL = root.appending(path: "artifacts/hf/MiniCPM5-1B/tokenizer.json")
    guard FileManager.default.fileExists(atPath: prefillURL.path),
          FileManager.default.fileExists(atPath: decodeURL.path),
          FileManager.default.fileExists(atPath: tokenizerURL.path)
    else {
        return
    }

    let bundle = CoreMLPrefillDecodeBundle.miniCPMExplicitKV(
        prefillModelURL: prefillURL,
        decodeModelURL: decodeURL,
        maxPromptTokens: 16
    )
    let report = try CoreMLPrefillDecodeDiagnostics(
        bundle: bundle,
        tokenizer: try MiniCPMBytePairTokenizer(tokenizerJSONURL: tokenizerURL, addBosToken: true)
    ).run(
        prompt: "Explain in one short paragraph why a split prefill/decode graph helps watch inference.",
        topK: 5
    )

    #expect(report.prefillTopK.count == 5)
    #expect(report.decodeTopK.count == 5)
    print("WATCHLM_REAL_INT8_DIAGNOSTIC en-short-001 prefill=\(report.prefillTopK.map(\.tokenID)) decode=\(report.decodeTopK.map(\.tokenID))")
}

@Test func coreMLPrefillDecodeDiagnosticsCanCompareLocalMiniCPMPrefillPrecisionArtifacts() throws {
    guard ProcessInfo.processInfo.environment["WATCHLM_RUN_REAL_COREML_TESTS"] == "1" else {
        return
    }

    let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let tokenizerURL = root.appending(path: "artifacts/hf/MiniCPM5-1B/tokenizer.json")
    let fp16PrefillURL = root.appending(path: "artifacts/coreml/real-minicpm5-prefill-kv-16/prefill-kv-16.mlpackage")
    let fp16DecodeURL = root.appending(path: "artifacts/coreml/real-minicpm5-decode-16/decode-16.mlpackage")
    let int8PrefillURL = root.appending(path: "artifacts/coreml/real-minicpm5-prefill-kv-16-int8/prefill-kv-16-int8.mlpackage")
    let int8DecodeURL = root.appending(path: "artifacts/coreml/real-minicpm5-decode-16-int8/decode-16-int8.mlpackage")
    guard [
        tokenizerURL,
        fp16PrefillURL,
        fp16DecodeURL,
        int8PrefillURL,
        int8DecodeURL
    ].allSatisfy({ FileManager.default.fileExists(atPath: $0.path) }) else {
        return
    }

    let prompt = "Explain in one short paragraph why a split prefill/decode graph helps watch inference."
    let tokenizer = try MiniCPMBytePairTokenizer(tokenizerJSONURL: tokenizerURL, addBosToken: true)
    let artifacts: [(id: String, prefillURL: URL, decodeURL: URL)] = [
        ("fp16-prefill-fp16-decode", fp16PrefillURL, fp16DecodeURL),
        ("fp16-prefill-int8-decode", fp16PrefillURL, int8DecodeURL),
        ("int8-prefill-fp16-decode", int8PrefillURL, fp16DecodeURL),
        ("int8-prefill-int8-decode", int8PrefillURL, int8DecodeURL)
    ]

    var summaries: [String] = []
    for artifact in artifacts {
        let bundle = CoreMLPrefillDecodeBundle.miniCPMExplicitKV(
            prefillModelURL: artifact.prefillURL,
            decodeModelURL: artifact.decodeURL,
            maxPromptTokens: 16
        )
        let report = try CoreMLPrefillDecodeDiagnostics(
            bundle: bundle,
            tokenizer: tokenizer
        ).run(prompt: prompt, topK: 5)

        #expect(report.prefillTopK.count == 5)
        #expect(report.decodeTopK.count == 5)
        summaries.append(
            "\(artifact.id) prefill=\(report.prefillTopK.map(\.tokenID)) decode=\(report.decodeTopK.map(\.tokenID)) decodeMargin=\(formattedTop1Margin(report.decodeTopK))"
        )
    }

    print("WATCHLM_PREFILL_PRECISION_DIAGNOSTIC \(summaries.joined(separator: " | "))")
}

@Test func coreMLPrefillDecodeDiagnosticsCanRunLocalMiniCPMPrefillProtectedArtifacts() throws {
    guard ProcessInfo.processInfo.environment["WATCHLM_RUN_REAL_COREML_TESTS"] == "1" else {
        return
    }

    let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let tokenizerURL = root.appending(path: "artifacts/hf/MiniCPM5-1B/tokenizer.json")
    let protectedPrefillURL = root.appending(path: "artifacts/coreml/real-minicpm5-prefill-kv-16-prefill-protected/prefill-kv-16-mixed.mlpackage")
    let fp16DecodeURL = root.appending(path: "artifacts/coreml/real-minicpm5-decode-16/decode-16.mlpackage")
    let int8DecodeURL = root.appending(path: "artifacts/coreml/real-minicpm5-decode-16-int8/decode-16-int8.mlpackage")
    guard [
        tokenizerURL,
        protectedPrefillURL,
        fp16DecodeURL,
        int8DecodeURL
    ].allSatisfy({ FileManager.default.fileExists(atPath: $0.path) }) else {
        return
    }

    let prompt = "Explain in one short paragraph why a split prefill/decode graph helps watch inference."
    let tokenizer = try MiniCPMBytePairTokenizer(tokenizerJSONURL: tokenizerURL, addBosToken: true)
    let artifacts: [(id: String, decodeURL: URL)] = [
        ("protected-prefill-fp16-decode", fp16DecodeURL),
        ("protected-prefill-int8-decode", int8DecodeURL)
    ]

    var summaries: [String] = []
    for artifact in artifacts {
        let bundle = CoreMLPrefillDecodeBundle.miniCPMExplicitKV(
            prefillModelURL: protectedPrefillURL,
            decodeModelURL: artifact.decodeURL,
            maxPromptTokens: 16
        )
        let report = try CoreMLPrefillDecodeDiagnostics(
            bundle: bundle,
            tokenizer: tokenizer
        ).run(prompt: prompt, topK: 5)

        #expect(report.prefillTopK.count == 5)
        #expect(report.decodeTopK.count == 5)
        #expect(report.firstDecodeTokenID == 4245)
        summaries.append(
            "\(artifact.id) prefill=\(report.prefillTopK.map(\.tokenID)) decode=\(report.decodeTopK.map(\.tokenID)) decodeMargin=\(formattedTop1Margin(report.decodeTopK))"
        )
    }

    print("WATCHLM_PREFILL_PROTECTED_DIAGNOSTIC \(summaries.joined(separator: " | "))")
}

@Test func coreMLPrefillDecodeRuntimeCanRunLocalRealMiniCPMFFN1013MixedArtifacts() async throws {
    guard ProcessInfo.processInfo.environment["WATCHLM_RUN_REAL_COREML_TESTS"] == "1" else {
        return
    }

    let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let prefillURL = root.appending(path: "artifacts/coreml/real-minicpm5-prefill-kv-16-mixed-ffn10-13/prefill-kv-16-mixed.mlpackage")
    let decodeURL = root.appending(path: "artifacts/coreml/real-minicpm5-decode-16-mixed-ffn10-13/decode-16-mixed.mlpackage")
    let tokenizerURL = root.appending(path: "artifacts/hf/MiniCPM5-1B/tokenizer.json")
    guard FileManager.default.fileExists(atPath: prefillURL.path),
          FileManager.default.fileExists(atPath: decodeURL.path),
          FileManager.default.fileExists(atPath: tokenizerURL.path)
    else {
        return
    }

    let bundle = CoreMLPrefillDecodeBundle.miniCPMExplicitKV(
        prefillModelURL: prefillURL,
        decodeModelURL: decodeURL,
        maxPromptTokens: 16
    )
    let runtime = CoreMLPrefillDecodeRuntime(
        bundle: bundle,
        tokenizer: try MiniCPMBytePairTokenizer(tokenizerJSONURL: tokenizerURL, addBosToken: true)
    )
    let artifact = RuntimeBenchmarkArtifact(
        quantizationPolicyID: "mixed-int4-ffn10-13-int8-rest",
        graphInterface: "logits-layered-kv",
        prefillModelPath: "artifacts/coreml/real-minicpm5-prefill-kv-16-mixed-ffn10-13/prefill-kv-16-mixed.mlpackage",
        decodeModelPath: "artifacts/coreml/real-minicpm5-decode-16-mixed-ffn10-13/decode-16-mixed.mlpackage",
        tokenizerPath: "artifacts/hf/MiniCPM5-1B/tokenizer.json"
    )
    let report = try await RuntimeBenchmarkRunner().run(
        runtime: runtime,
        configuration: RuntimeBenchmarkConfiguration(
            id: "real-minicpm5-context16-mixed-ffn10-13-local",
            sourceModelId: "openbmb/MiniCPM5-1B",
            runtime: "coreml-mlprogram",
            deviceProfile: .watchSE2,
            contextVariant: 16,
            artifact: artifact
        ),
        prompts: [
            RuntimeBenchmarkPrompt(
                id: "apple-watch-local-inference-test",
                category: "watch_utility",
                language: "en",
                input: "Apple Watch local inference test.",
                maxNewTokens: 2,
                qualityReference: RuntimeQualityReference(
                    source: "pytorch-teacher-context16-ffn10-13-validation",
                    tokenIDs: [242, 38]
                )
            )
        ]
    )
    let result = try #require(report.promptResults.first)

    #expect(report.configuration.artifact == artifact)
    #expect(report.summary.succeededPromptCount == 1)
    #expect(report.summary.averageTokenAgreement == 1.0)
    #expect(result.generatedTokenIDs == [242, 38])
    #expect(result.timing.decodeStepMs.count == 1)
    #expect(result.metrics.kvCacheUpdateStrategy == .slotRing)
}

@Test func coreMLPrefillDecodeRuntimeCanRunLocalRealMiniCPMFFN815MixedArtifacts() async throws {
    guard ProcessInfo.processInfo.environment["WATCHLM_RUN_REAL_COREML_TESTS"] == "1" else {
        return
    }

    let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let prefillURL = root.appending(path: "artifacts/coreml/real-minicpm5-prefill-kv-16-mixed-ffn8-15/prefill-kv-16-mixed.mlpackage")
    let decodeURL = root.appending(path: "artifacts/coreml/real-minicpm5-decode-16-mixed-ffn8-15/decode-16-mixed.mlpackage")
    let tokenizerURL = root.appending(path: "artifacts/hf/MiniCPM5-1B/tokenizer.json")
    guard FileManager.default.fileExists(atPath: prefillURL.path),
          FileManager.default.fileExists(atPath: decodeURL.path),
          FileManager.default.fileExists(atPath: tokenizerURL.path)
    else {
        return
    }

    let bundle = CoreMLPrefillDecodeBundle.miniCPMExplicitKV(
        prefillModelURL: prefillURL,
        decodeModelURL: decodeURL,
        maxPromptTokens: 16
    )
    let runtime = CoreMLPrefillDecodeRuntime(
        bundle: bundle,
        tokenizer: try MiniCPMBytePairTokenizer(tokenizerJSONURL: tokenizerURL, addBosToken: true)
    )
    let artifact = RuntimeBenchmarkArtifact(
        quantizationPolicyID: "mixed-int4-ffn8-15-int8-rest",
        graphInterface: "logits-layered-kv",
        prefillModelPath: "artifacts/coreml/real-minicpm5-prefill-kv-16-mixed-ffn8-15/prefill-kv-16-mixed.mlpackage",
        decodeModelPath: "artifacts/coreml/real-minicpm5-decode-16-mixed-ffn8-15/decode-16-mixed.mlpackage",
        tokenizerPath: "artifacts/hf/MiniCPM5-1B/tokenizer.json"
    )
    let report = try await RuntimeBenchmarkRunner().run(
        runtime: runtime,
        configuration: RuntimeBenchmarkConfiguration(
            id: "real-minicpm5-context16-mixed-ffn8-15-local",
            sourceModelId: "openbmb/MiniCPM5-1B",
            runtime: "coreml-mlprogram",
            deviceProfile: .watchSE2,
            contextVariant: 16,
            artifact: artifact
        ),
        prompts: [
            RuntimeBenchmarkPrompt(
                id: "apple-watch-local-inference-test",
                category: "watch_utility",
                language: "en",
                input: "Apple Watch local inference test.",
                maxNewTokens: 2,
                qualityReference: RuntimeQualityReference(
                    source: "pytorch-teacher-context16-ffn8-15-validation",
                    tokenIDs: [242, 38]
                )
            )
        ]
    )
    let result = try #require(report.promptResults.first)

    #expect(report.configuration.artifact == artifact)
    #expect(report.summary.succeededPromptCount == 1)
    #expect(report.summary.averageTokenAgreement == 1.0)
    #expect(result.generatedTokenIDs == [242, 38])
    #expect(result.timing.decodeStepMs.count == 1)
    #expect(result.metrics.kvCacheUpdateStrategy == .slotRing)
}
#endif

@Test func coreMLPrefillDecodeRuntimeStreamsLayeredKVTokenEvents() async throws {
    let prefillURL = try #require(smokeModelURL(named: "SmokeLayeredPrefill"))
    let decodeURL = try #require(smokeModelURL(named: "SmokeLayeredDecode"))
    let bundle = CoreMLPrefillDecodeBundle(
        prefillModelURL: prefillURL,
        decodeModelURL: decodeURL,
        maxPromptTokens: 4,
        graphInterface: .logitsAndLayeredKV(layerCount: 1, kvHeads: 1, headDimension: 1),
        decodeTokenInputName: "token_id"
    )
    let runtime = CoreMLPrefillDecodeRuntime(
        bundle: bundle,
        tokenizer: FixtureTokenIDTokenizer()
    )

    var events: [InferenceStreamEvent] = []
    for try await event in runtime.stream(
        request: InferenceRequest(prompt: "A B", maxNewTokens: 2),
        shouldCancel: { false }
    ) {
        events.append(event)
    }

    #expect(events.count == 3)
    #expect(events[0] == .token(InferenceToken(index: 0, tokenID: 5, text: "D", isFirstToken: true)))
    #expect(events[1] == .token(InferenceToken(index: 1, tokenID: 6, text: "E", isFirstToken: false)))
    if case .completed(let result) = events[2] {
        #expect(result.generatedTokenIDs == [5, 6])
        #expect(result.text == "DE")
        #expect(result.terminationReason == .maxTokens)
    } else {
        Issue.record("Expected stream completion event")
    }
}

@Test func coreMLPrefillDecodeRuntimeReportsEndOfSequenceFromPrefillToken() async throws {
    let prefillURL = try #require(smokeModelURL(named: "SmokeLayeredPrefill"))
    let decodeURL = try #require(smokeModelURL(named: "SmokeLayeredDecode"))
    let bundle = CoreMLPrefillDecodeBundle(
        prefillModelURL: prefillURL,
        decodeModelURL: decodeURL,
        maxPromptTokens: 4,
        graphInterface: .logitsAndLayeredKV(layerCount: 1, kvHeads: 1, headDimension: 1),
        decodeTokenInputName: "token_id"
    )
    let runtime = CoreMLPrefillDecodeRuntime(
        bundle: bundle,
        tokenizer: FixtureTokenIDTokenizer(endOfSequenceTokenIDs: [5])
    )

    let result = try await runtime.generate(
        request: InferenceRequest(prompt: "A B", maxNewTokens: 3),
        shouldCancel: { false }
    )

    #expect(result.tokens == [])
    #expect(result.generatedTokenIDs == [])
    #expect(result.terminationReason == .endOfSequence)
    #expect(result.timing.decodeStepMs.isEmpty)
}

@Test func coreMLPrefillDecodeRuntimeLoadPreservesGraphIOMismatchDiagnostics() async throws {
    let prefillURL = try #require(smokeModelURL(named: "SmokeLayeredPrefill"))
    let decodeURL = try #require(smokeModelURL(named: "SmokeLayeredDecode"))
    let bundle = CoreMLPrefillDecodeBundle(
        prefillModelURL: prefillURL,
        decodeModelURL: decodeURL,
        maxPromptTokens: 4,
        graphInterface: .logitsAndLayeredKV(layerCount: 2, kvHeads: 1, headDimension: 1),
        decodeTokenInputName: "token_id"
    )
    let runtime = CoreMLPrefillDecodeRuntime(
        bundle: bundle,
        tokenizer: FixtureTokenIDTokenizer()
    )

    do {
        _ = try await runtime.load()
        Issue.record("Expected runtime load to reject mismatched graph IO")
    } catch let error as InferenceRuntimeError {
        #expect(error.userMessage.contains("decode outputs: new_key_1, new_value_1"))
    }
}

@Test func miniCPMExplicitKVBundleUsesRealGraphFeatureNames() throws {
    let bundle = CoreMLPrefillDecodeBundle.miniCPMExplicitKV(
        prefillModelURL: URL(fileURLWithPath: "/tmp/prefill.mlpackage"),
        decodeModelURL: URL(fileURLWithPath: "/tmp/decode.mlpackage"),
        maxPromptTokens: 16
    )

    #expect(bundle.graphInterface == .logitsAndLayeredKV(layerCount: 24, kvHeads: 2, headDimension: 128))
    #expect(bundle.prefillInputName == "input_ids")
    #expect(bundle.prefillPositionInputName == "position_ids")
    #expect(bundle.prefillCausalMaskInputName == "causal_mask")
    #expect(bundle.prefillLogitsOutputName == "logits")
    #expect(bundle.prefillKeyOutputName(forLayer: 3) == "present_key_3")
    #expect(bundle.prefillValueOutputName(forLayer: 3) == "present_value_3")
    #expect(bundle.decodeTokenInputName == "token_id")
    #expect(bundle.decodePositionInputName == "position_id")
    #expect(bundle.decodeCausalMaskInputName == "causal_mask")
    #expect(bundle.decodePastKeyInputName(forLayer: 3) == "past_key_3")
    #expect(bundle.decodePastValueInputName(forLayer: 3) == "past_value_3")
    #expect(bundle.decodeNewKeyOutputName(forLayer: 3) == "new_key_3")
    #expect(bundle.decodeNewValueOutputName(forLayer: 3) == "new_value_3")
}

@Test func coreMLPrefillDecodeBundleCarriesLogitsProcessingPolicy() throws {
    let bundle = CoreMLPrefillDecodeBundle(
        prefillModelURL: URL(fileURLWithPath: "/tmp/prefill.mlpackage"),
        decodeModelURL: URL(fileURLWithPath: "/tmp/decode.mlpackage"),
        maxPromptTokens: 16,
        logitsProcessor: LogitsProcessor(temperature: 0.8, topK: 20, topP: 0.9, repetitionPenalty: 1.1)
    )

    #expect(bundle.logitsProcessor.temperature == 0.8)
    #expect(bundle.logitsProcessor.topK == 20)
    #expect(bundle.logitsProcessor.topP == 0.9)
    #expect(bundle.logitsProcessor.repetitionPenalty == 1.1)
}

@Test func coreMLPrefillDecodeBundleCarriesSamplingStrategy() throws {
    let bundle = CoreMLPrefillDecodeBundle(
        prefillModelURL: URL(fileURLWithPath: "/tmp/prefill.mlpackage"),
        decodeModelURL: URL(fileURLWithPath: "/tmp/decode.mlpackage"),
        maxPromptTokens: 16,
        samplingStrategy: .seeded(seed: 123)
    )

    #expect(bundle.samplingStrategy == .seeded(seed: 123))
}

@Test func coreMLPrefillDecodeBundleRejectsGraphIOMismatches() throws {
    let bundle = CoreMLPrefillDecodeBundle(
        prefillModelURL: URL(fileURLWithPath: "/tmp/prefill.mlpackage"),
        decodeModelURL: URL(fileURLWithPath: "/tmp/decode.mlpackage"),
        maxPromptTokens: 16,
        graphInterface: .logitsAndLayeredKV(layerCount: 2, kvHeads: 2, headDimension: 128),
        decodeTokenInputName: "token_id"
    )

    do {
        try bundle.validateGraphIOContract(
            prefillInputNames: ["input_ids", "position_ids", "causal_mask"],
            prefillOutputNames: ["logits", "present_key_0", "present_value_0", "present_key_1", "present_value_1"],
            decodeInputNames: [
                "token_id", "position_id", "causal_mask",
                "past_key_0", "past_value_0",
                "past_key_1", "past_value_1"
            ],
            decodeOutputNames: ["logits", "new_key_0", "new_value_0", "new_key_1"]
        )
        Issue.record("Expected graph IO contract validation to reject missing decode output")
    } catch let error as InferenceRuntimeError {
        #expect(error.userMessage.contains("decode outputs: new_value_1"))
    }
}

@Test func coreMLPrefillDecodeBundleRejectsGraphIOShapeMismatches() throws {
    let bundle = CoreMLPrefillDecodeBundle(
        prefillModelURL: URL(fileURLWithPath: "/tmp/prefill.mlpackage"),
        decodeModelURL: URL(fileURLWithPath: "/tmp/decode.mlpackage"),
        maxPromptTokens: 4,
        graphInterface: .logitsAndLayeredKV(layerCount: 1, kvHeads: 2, headDimension: 128),
        decodeTokenInputName: "token_id"
    )

    do {
        try bundle.validateGraphIOContract(
            prefillInputShapes: [
                "input_ids": [1, 4],
                "position_ids": [1, 4],
                "causal_mask": [1, 1, 4, 4]
            ],
            prefillOutputShapes: [
                "logits": [1, 130_072],
                "present_key_0": [1, 2, 4, 128],
                "present_value_0": [1, 2, 4, 128]
            ],
            decodeInputShapes: [
                "token_id": [1, 1],
                "position_id": [1, 1],
                "causal_mask": [1, 1, 1, 5],
                "past_key_0": [1, 2, 4, 128],
                "past_value_0": [1, 2, 4, 128]
            ],
            decodeOutputShapes: [
                "logits": [1, 130_072],
                "new_key_0": [1, 2, 1, 64],
                "new_value_0": [1, 2, 1, 128]
            ]
        )
        Issue.record("Expected graph IO contract validation to reject wrong new_key shape")
    } catch let error as InferenceRuntimeError {
        #expect(error.userMessage.contains("decode outputs new_key_0 shape [1, 2, 1, 64] expected [1, 2, 1, 128]"))
    }
}

@Test func coreMLPrefillDecodeBundleRejectsVectorPrefillInputShapesForLayeredKVGraphs() throws {
    let bundle = CoreMLPrefillDecodeBundle(
        prefillModelURL: URL(fileURLWithPath: "/tmp/prefill.mlpackage"),
        decodeModelURL: URL(fileURLWithPath: "/tmp/decode.mlpackage"),
        maxPromptTokens: 4,
        graphInterface: .logitsAndLayeredKV(layerCount: 1, kvHeads: 2, headDimension: 128),
        decodeTokenInputName: "token_id"
    )

    do {
        try bundle.validateGraphIOContract(
            prefillInputShapes: [
                "input_ids": [4],
                "position_ids": [4],
                "causal_mask": [1, 1, 4, 4]
            ],
            prefillOutputShapes: [
                "logits": [1, 130_072],
                "present_key_0": [1, 2, 4, 128],
                "present_value_0": [1, 2, 4, 128]
            ],
            decodeInputShapes: [
                "token_id": [1, 1],
                "position_id": [1, 1],
                "causal_mask": [1, 1, 1, 5],
                "past_key_0": [1, 2, 4, 128],
                "past_value_0": [1, 2, 4, 128]
            ],
            decodeOutputShapes: [
                "logits": [1, 130_072],
                "new_key_0": [1, 2, 1, 128],
                "new_value_0": [1, 2, 1, 128]
            ]
        )
        Issue.record("Expected graph IO contract validation to reject vector prefill inputs")
    } catch let error as InferenceRuntimeError {
        #expect(error.userMessage.contains("prefill inputs input_ids shape [4] expected [1, 4]"))
        #expect(error.userMessage.contains("position_ids shape [4] expected [1, 4]"))
    }
}

@Test func coreMLPrefillDecodeBundleRejectsGraphIODTypeMismatches() throws {
    let bundle = CoreMLPrefillDecodeBundle(
        prefillModelURL: URL(fileURLWithPath: "/tmp/prefill.mlpackage"),
        decodeModelURL: URL(fileURLWithPath: "/tmp/decode.mlpackage"),
        maxPromptTokens: 4,
        graphInterface: .logitsAndLayeredKV(layerCount: 1, kvHeads: 2, headDimension: 128),
        decodeTokenInputName: "token_id"
    )

    do {
        try bundle.validateGraphIOContract(
            prefillInputDataTypes: [
                "input_ids": .float16,
                "position_ids": .int32,
                "causal_mask": .float16
            ],
            prefillOutputDataTypes: [
                "logits": .float16,
                "present_key_0": .float16,
                "present_value_0": .float16
            ],
            decodeInputDataTypes: [
                "token_id": .int32,
                "position_id": .int32,
                "causal_mask": .float16,
                "past_key_0": .float16,
                "past_value_0": .float16
            ],
            decodeOutputDataTypes: [
                "logits": .float16,
                "new_key_0": .float16,
                "new_value_0": .float16
            ]
        )
        Issue.record("Expected graph IO contract validation to reject wrong input_ids dtype")
    } catch let error as InferenceRuntimeError {
        #expect(error.userMessage.contains("prefill inputs input_ids dtype float16 expected int32"))
    }
}

@Test func coreMLPrefillDecodeBundleAcceptsSmokeModelGraphIOShapes() async throws {
    let prefillURL = try #require(smokeModelURL(named: "SmokeLayeredPrefill"))
    let decodeURL = try #require(smokeModelURL(named: "SmokeLayeredDecode"))
    let prefill = try MLModel(contentsOf: prefillURL)
    let decode = try MLModel(contentsOf: decodeURL)
    let bundle = CoreMLPrefillDecodeBundle(
        prefillModelURL: prefillURL,
        decodeModelURL: decodeURL,
        maxPromptTokens: 4,
        graphInterface: .logitsAndLayeredKV(layerCount: 1, kvHeads: 1, headDimension: 1),
        decodeTokenInputName: "token_id"
    )

    try bundle.validateModelDescriptions(
        prefill: prefill.modelDescription,
        decode: decode.modelDescription
    )
}

@Test func coreMLLogitsSamplerCanUseSeededProbabilitySampler() throws {
    let logits = try multiArray(shape: [1, 4], values: [0, 0, 0, 0])
    let first = CoreMLLogitsSampler(sampler: TokenSamplingStrategy.seeded(seed: 7).makeSampler())
    let second = CoreMLLogitsSampler(sampler: TokenSamplingStrategy.seeded(seed: 7).makeSampler())

    let firstTokens = try (0..<10).map { _ in try first.selectToken(from: logits) }
    let secondTokens = try (0..<10).map { _ in try second.selectToken(from: logits) }

    #expect(firstTokens == secondTokens)
    #expect(Set(firstTokens).count > 1)
}

@Test func kvTensorLayoutDescribesMiniCPMCoreMLKVShapes() {
    let layout = KVTensorLayout(
        batchSize: 1,
        kvHeads: 2,
        contextTokens: 256,
        headDimension: 128
    )
    let descriptor = KVCacheDescriptor.miniCPM5(contextTokens: 256, precision: .int8)

    #expect(layout.tensorShape == [1, 2, 256, 128])
    #expect(layout.decodeSliceShape == [1, 2, 1, 128])
    #expect(layout.scalarCountPerTensor == 65_536)
    #expect(layout.byteCount(layerCount: 24, precision: .int8) == descriptor.totalBytes)
}

@Test func coreMLKVCacheStoreAppendsDecodeOutputsIntoSlidingPastWindow() throws {
    let key = try multiArray(shape: [1, 1, 3, 1], values: [10, 11, 12])
    let value = try multiArray(shape: [1, 1, 3, 1], values: [20, 21, 22])
    let output = FixtureFeatureProvider(features: [
        "present_key_0": MLFeatureValue(multiArray: key),
        "present_value_0": MLFeatureValue(multiArray: value)
    ])
    var cache = try CoreMLKVCacheStore(
        prefillOutput: output,
        layerCount: 1,
        keyOutputName: { "present_key_\($0)" },
        valueOutputName: { "present_value_\($0)" }
    )

    let newKey = try multiArray(shape: [1, 1, 1, 1], values: [99])
    let newValue = try multiArray(shape: [1, 1, 1, 1], values: [199])
    let decodeOutput = FixtureFeatureProvider(features: [
        "new_key_0": MLFeatureValue(multiArray: newKey),
        "new_value_0": MLFeatureValue(multiArray: newValue)
    ])

    try cache.appendDecodeOutputs(
        output: decodeOutput,
        keyOutputName: { "new_key_\($0)" },
        valueOutputName: { "new_value_\($0)" }
    )

    #expect(cache.key(forLayer: 0)[0].doubleValue == 11)
    #expect(cache.key(forLayer: 0)[1].doubleValue == 12)
    #expect(cache.key(forLayer: 0)[2].doubleValue == 99)
    #expect(cache.value(forLayer: 0)[0].doubleValue == 21)
    #expect(cache.value(forLayer: 0)[1].doubleValue == 22)
    #expect(cache.value(forLayer: 0)[2].doubleValue == 199)
    #expect(cache.layout == KVTensorLayout(batchSize: 1, kvHeads: 1, contextTokens: 3, headDimension: 1))
}

@Test func coreMLKVCacheStoreUsesActiveWindowBeforeSlidingFullContext() throws {
    let key = try multiArray(shape: [1, 1, 4, 1], values: [0, 0, 10, 11])
    let value = try multiArray(shape: [1, 1, 4, 1], values: [0, 0, 20, 21])
    let output = FixtureFeatureProvider(features: [
        "present_key_0": MLFeatureValue(multiArray: key),
        "present_value_0": MLFeatureValue(multiArray: value)
    ])
    var cache = try CoreMLKVCacheStore(
        prefillOutput: output,
        layerCount: 1,
        validTokenCount: 2,
        keyOutputName: { "present_key_\($0)" },
        valueOutputName: { "present_value_\($0)" }
    )

    let firstDecodeOutput = FixtureFeatureProvider(features: [
        "new_key_0": MLFeatureValue(multiArray: try multiArray(shape: [1, 1, 1, 1], values: [99])),
        "new_value_0": MLFeatureValue(multiArray: try multiArray(shape: [1, 1, 1, 1], values: [199]))
    ])
    try cache.appendDecodeOutputs(
        output: firstDecodeOutput,
        keyOutputName: { "new_key_\($0)" },
        valueOutputName: { "new_value_\($0)" }
    )

    #expect(cache.activeTokenStartIndex == 1)
    #expect(cache.validTokenCount == 3)
    #expect(cache.lastAppendMovedTokenCount == 2)
    #expect(cache.lastAppendMovedScalarCount == 4)
    #expect(cache.key(forLayer: 0)[0].doubleValue == 0)
    #expect(cache.key(forLayer: 0)[1].doubleValue == 10)
    #expect(cache.key(forLayer: 0)[2].doubleValue == 11)
    #expect(cache.key(forLayer: 0)[3].doubleValue == 99)

    let secondDecodeOutput = FixtureFeatureProvider(features: [
        "new_key_0": MLFeatureValue(multiArray: try multiArray(shape: [1, 1, 1, 1], values: [100])),
        "new_value_0": MLFeatureValue(multiArray: try multiArray(shape: [1, 1, 1, 1], values: [200]))
    ])
    try cache.appendDecodeOutputs(
        output: secondDecodeOutput,
        keyOutputName: { "new_key_\($0)" },
        valueOutputName: { "new_value_\($0)" }
    )

    #expect(cache.activeTokenStartIndex == 0)
    #expect(cache.validTokenCount == 4)
    #expect(cache.lastAppendMovedTokenCount == 3)
    #expect(cache.lastAppendMovedScalarCount == 6)
    #expect(cache.key(forLayer: 0)[0].doubleValue == 10)
    #expect(cache.key(forLayer: 0)[1].doubleValue == 11)
    #expect(cache.key(forLayer: 0)[2].doubleValue == 99)
    #expect(cache.key(forLayer: 0)[3].doubleValue == 100)
}

@Test func coreMLKVCacheStoreCanUseSlotRingWithoutMovingPastTokens() throws {
    let key = try multiArray(shape: [1, 1, 4, 1], values: [0, 0, 10, 11])
    let value = try multiArray(shape: [1, 1, 4, 1], values: [0, 0, 20, 21])
    let output = FixtureFeatureProvider(features: [
        "present_key_0": MLFeatureValue(multiArray: key),
        "present_value_0": MLFeatureValue(multiArray: value)
    ])
    var cache = try CoreMLKVCacheStore(
        prefillOutput: output,
        layerCount: 1,
        validTokenCount: 2,
        updateStrategy: .slotRing,
        keyOutputName: { "present_key_\($0)" },
        valueOutputName: { "present_value_\($0)" }
    )

    let firstDecodeOutput = FixtureFeatureProvider(features: [
        "new_key_0": MLFeatureValue(multiArray: try multiArray(shape: [1, 1, 1, 1], values: [99])),
        "new_value_0": MLFeatureValue(multiArray: try multiArray(shape: [1, 1, 1, 1], values: [199]))
    ])
    try cache.appendDecodeOutputs(
        output: firstDecodeOutput,
        keyOutputName: { "new_key_\($0)" },
        valueOutputName: { "new_value_\($0)" }
    )

    #expect(cache.activeTokenStartIndex == 1)
    #expect(cache.validTokenCount == 3)
    #expect(cache.lastAppendWriteIndex == 1)
    #expect(cache.lastAppendMovedTokenCount == 0)
    #expect(cache.lastAppendMovedScalarCount == 0)
    #expect(cache.key(forLayer: 0)[0].doubleValue == 0)
    #expect(cache.key(forLayer: 0)[1].doubleValue == 99)
    #expect(cache.key(forLayer: 0)[2].doubleValue == 10)
    #expect(cache.key(forLayer: 0)[3].doubleValue == 11)

    let secondDecodeOutput = FixtureFeatureProvider(features: [
        "new_key_0": MLFeatureValue(multiArray: try multiArray(shape: [1, 1, 1, 1], values: [100])),
        "new_value_0": MLFeatureValue(multiArray: try multiArray(shape: [1, 1, 1, 1], values: [200]))
    ])
    try cache.appendDecodeOutputs(
        output: secondDecodeOutput,
        keyOutputName: { "new_key_\($0)" },
        valueOutputName: { "new_value_\($0)" }
    )

    #expect(cache.activeTokenStartIndex == 0)
    #expect(cache.validTokenCount == 4)
    #expect(cache.lastAppendWriteIndex == 0)
    #expect(cache.lastAppendMovedTokenCount == 0)
    #expect(cache.lastAppendMovedScalarCount == 0)
    #expect(cache.key(forLayer: 0)[0].doubleValue == 100)
    #expect(cache.key(forLayer: 0)[1].doubleValue == 99)
    #expect(cache.key(forLayer: 0)[2].doubleValue == 10)
    #expect(cache.key(forLayer: 0)[3].doubleValue == 11)

    let thirdDecodeOutput = FixtureFeatureProvider(features: [
        "new_key_0": MLFeatureValue(multiArray: try multiArray(shape: [1, 1, 1, 1], values: [101])),
        "new_value_0": MLFeatureValue(multiArray: try multiArray(shape: [1, 1, 1, 1], values: [201]))
    ])
    try cache.appendDecodeOutputs(
        output: thirdDecodeOutput,
        keyOutputName: { "new_key_\($0)" },
        valueOutputName: { "new_value_\($0)" }
    )

    #expect(cache.lastAppendWriteIndex == 2)
    #expect(cache.lastAppendMovedTokenCount == 0)
    #expect(cache.lastAppendMovedScalarCount == 0)
    #expect(cache.key(forLayer: 0)[0].doubleValue == 100)
    #expect(cache.key(forLayer: 0)[1].doubleValue == 99)
    #expect(cache.key(forLayer: 0)[2].doubleValue == 101)
    #expect(cache.key(forLayer: 0)[3].doubleValue == 11)
    #expect(cache.value(forLayer: 0)[2].doubleValue == 201)
}

@Test func coreMLKVCacheStoreRejectsMismatchedLayerShapes() throws {
    let key = try multiArray(shape: [1, 1, 3, 1], values: [10, 11, 12])
    let value = try multiArray(shape: [1, 2, 3, 1], values: [20, 21, 22, 23, 24, 25])
    let output = FixtureFeatureProvider(features: [
        "present_key_0": MLFeatureValue(multiArray: key),
        "present_value_0": MLFeatureValue(multiArray: value)
    ])

    do {
        _ = try CoreMLKVCacheStore(
            prefillOutput: output,
            layerCount: 1,
            keyOutputName: { "present_key_\($0)" },
            valueOutputName: { "present_value_\($0)" }
        )
        Issue.record("Expected KV layout validation to reject mismatched key/value shapes")
    } catch let error as InferenceRuntimeError {
        #expect(error.userMessage.contains("Prediction failed"))
    }
}

@Test func coreMLLogitsGreedySamplerSelectsHighestTokenAndLowestTie() throws {
    let logits = try multiArray(shape: [1, 5], values: [0.1, 3.0, 1.5, 3.0, -1.0])

    let selected = try CoreMLLogitsGreedySampler().selectToken(from: logits)

    #expect(selected == 1)
}

@Test func coreMLLogitsProcessorExtractsTopKTokenLogitsFromMLMultiArray() throws {
    let logits = try multiArray(shape: [1, 5], values: [0.1, 3.0, 1.5, 3.0, -1.0])

    let tokenLogits = try CoreMLLogitsProcessor(topK: 3).tokenLogits(from: logits)

    #expect(tokenLogits == [
        TokenLogit(tokenID: 1, logit: 3.0),
        TokenLogit(tokenID: 3, logit: 3.0),
        TokenLogit(tokenID: 2, logit: 1.5)
    ])
}

@Test func coreMLLogitsProcessorAppliesSharedDecodePolicyToMLMultiArray() throws {
    let logits = try multiArray(shape: [1, 4], values: [8.0, 6.0, 4.0, 2.0])
    let processor = CoreMLLogitsProcessor(
        policy: LogitsProcessor(
            temperature: 2.0,
            topK: 3,
            topP: 0.78,
            repetitionPenalty: 2.0
        )
    )

    let tokenLogits = try processor.tokenLogits(from: logits, generatedTokenIDs: [0])

    #expect(tokenLogits == [
        TokenLogit(tokenID: 1, logit: 3.0),
        TokenLogit(tokenID: 0, logit: 2.0)
    ])
}

@Test func miniCPMPrefillInputStateMirrorsConversionPaddingPositionsAndMask() throws {
    let state = try CoreMLMiniCPMInputState(
        tokenIDs: [8, 9],
        capacity: 4,
        padTokenID: 1
    )

    #expect(state.realTokenCount == 2)
    #expect(state.inputIDs.shape.map(\.intValue) == [1, 4])
    #expect(state.positionIDs.shape.map(\.intValue) == [1, 4])
    #expect(state.inputIDs[[0, 0] as [NSNumber]].int32Value == 1)
    #expect(state.inputIDs[[0, 1] as [NSNumber]].int32Value == 1)
    #expect(state.inputIDs[[0, 2] as [NSNumber]].int32Value == 8)
    #expect(state.inputIDs[[0, 3] as [NSNumber]].int32Value == 9)
    #expect(state.positionIDs[[0, 0] as [NSNumber]].int32Value == 0)
    #expect(state.positionIDs[[0, 1] as [NSNumber]].int32Value == 0)
    #expect(state.positionIDs[[0, 2] as [NSNumber]].int32Value == 0)
    #expect(state.positionIDs[[0, 3] as [NSNumber]].int32Value == 1)
    #expect(state.causalMask[[0, 0, 3, 2] as [NSNumber]].doubleValue == 0)
    #expect(state.causalMask[[0, 0, 3, 1] as [NSNumber]].doubleValue == -65504)
}

@Test func miniCPMInputStateAdvancesDecodePositionAndMaskAfterGeneratedToken() throws {
    var state = try CoreMLMiniCPMInputState(
        tokenIDs: [8, 9],
        capacity: 4,
        padTokenID: 1
    )

    #expect(state.decodePositionID[0].int32Value == 2)
    #expect(state.decodeCausalMask[[0, 0, 0, 1] as [NSNumber]].doubleValue == -65504)
    #expect(state.decodeCausalMask[[0, 0, 0, 4] as [NSNumber]].doubleValue == 0)

    state.appendGeneratedToken()

    #expect(state.decodePositionID[0].int32Value == 3)
    #expect(state.decodeCausalMask[[0, 0, 0, 1] as [NSNumber]].doubleValue == 0)
}

@Test func miniCPMInputStateCanMarkGeneratedTokenAtRingKVSlot() throws {
    var state = try CoreMLMiniCPMInputState(
        tokenIDs: [8, 9],
        capacity: 4,
        padTokenID: 1
    )

    #expect(state.decodeCausalMask[[0, 0, 0, 1] as [NSNumber]].doubleValue == -65504)
    try state.appendGeneratedToken(atPastKVSlot: 1)

    #expect(state.decodePositionID[0].int32Value == 3)
    #expect(state.decodeCausalMask[[0, 0, 0, 0] as [NSNumber]].doubleValue == -65504)
    #expect(state.decodeCausalMask[[0, 0, 0, 1] as [NSNumber]].doubleValue == 0)
    #expect(state.decodeCausalMask[[0, 0, 0, 2] as [NSNumber]].doubleValue == 0)
    #expect(state.decodeCausalMask[[0, 0, 0, 3] as [NSNumber]].doubleValue == 0)

    try state.appendGeneratedToken(atPastKVSlot: 0)

    #expect(state.decodePositionID[0].int32Value == 4)
    #expect(state.decodeCausalMask[[0, 0, 0, 0] as [NSNumber]].doubleValue == 0)
}

private func smokeModelURL(named baseName: String) -> URL? {
    #if os(watchOS)
    Bundle.module.url(forResource: "\(baseName)_watchOS", withExtension: "mlmodelc")
    #else
    Bundle.module.url(forResource: "\(baseName)_macOS", withExtension: "mlmodelc")
    #endif
}

private func formattedTop1Margin(_ logits: [TokenLogit]) -> String {
    guard logits.count >= 2 else {
        return "n/a"
    }
    return String(format: "%.6f", logits[0].logit - logits[1].logit)
}

private struct FixtureTokenIDTokenizer: TextTokenizer {
    let endOfSequenceTokenIDs: Set<Int32>

    init(endOfSequenceTokenIDs: Set<Int32> = [1]) {
        self.endOfSequenceTokenIDs = endOfSequenceTokenIDs
    }

    func encode(_ text: String) throws -> [Int32] {
        try text
            .split(separator: " ")
            .map { piece in
                switch String(piece) {
                case "A": 2
                case "B": 3
                default:
                    throw InferenceRuntimeError.invalidInput(message: "Unknown fixture token \(piece).")
                }
            }
    }

    func decode(tokenIDs: [Int32]) throws -> String {
        try tokenIDs.map { tokenID in
            switch tokenID {
            case 5: "D"
            case 6: "E"
            case 7: "F"
            default:
                throw InferenceRuntimeError.predictionFailed(message: "Unknown fixture output token \(tokenID).")
            }
        }
        .joined()
    }
}

private final class FixtureFeatureProvider: MLFeatureProvider {
    private let features: [String: MLFeatureValue]

    var featureNames: Set<String> {
        Set(features.keys)
    }

    init(features: [String: MLFeatureValue]) {
        self.features = features
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        features[featureName]
    }
}

private func multiArray(shape: [Int], values: [Double]) throws -> MLMultiArray {
    let array = try MLMultiArray(
        shape: shape.map { NSNumber(value: $0) },
        dataType: .double
    )
    for (index, value) in values.enumerated() {
        array[index] = NSNumber(value: value)
    }
    return array
}
#endif
