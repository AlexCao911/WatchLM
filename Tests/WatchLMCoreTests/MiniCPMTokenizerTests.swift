import Foundation
import Testing
@testable import WatchLMCore

#if os(macOS)
@Test func miniCPMBytePairTokenizerMatchesLocalHFTokenizerSmoke() throws {
    let tokenizer = try MiniCPMBytePairTokenizer(
        tokenizerJSONURL: localMiniCPMTokenizerJSONURL(),
        addBosToken: true
    )

    #expect(try tokenizer.encode("Hello world!") == [0, 36417, 1782, 22])
    #expect(try tokenizer.encode("Answer briefly.") == [0, 21742, 15020, 35])
    #expect(try tokenizer.encode("你好") == [0, 75828])
    #expect(try tokenizer.decode(tokenIDs: [36417, 1782, 22]) == "Hello world!")
    #expect(try tokenizer.decode(tokenIDs: [0, 36417, 1782, 22]) == "<s>Hello world!")
}

@Test func bytePairTokenizerCanUseRuntimeCandidateSpecialTokens() throws {
    let tokenizer = try MiniCPMBytePairTokenizer(
        tokenizerJSONURL: localMiniCPMTokenizerJSONURL(),
        addBosToken: true,
        bosTokenID: 151643,
        eosTokenIDs: [151645]
    )

    #expect(tokenizer.endOfSequenceTokenIDs == [151645])
    #expect(try tokenizer.encode("Hello world!").first == 151643)
}

@Test func miniCPMBytePairTokenizerEncodesRenderedNoThinkTemplate() throws {
    let template = MiniCPMChatTemplate(bosToken: "<s>")
    let rendered = template.render(
        messages: [ChatMessage(role: .user, content: "Hi")],
        addGenerationPrompt: true,
        enableThinking: false
    )
    let tokenizer = try MiniCPMBytePairTokenizer(
        tokenizerJSONURL: localMiniCPMTokenizerJSONURL(),
        addBosToken: false
    )

    let tokenIDs = try tokenizer.encode(rendered)

    #expect(tokenIDs == [
        0, 130072, 8448, 220, 19301, 130073, 220,
        130072, 130071, 220, 8, 130063, 9, 130063
    ])
}

@Test func miniCPMBytePairTokenizerMatchesBenchmarkPromptSuiteHFReferences() throws {
    let tokenizer = try MiniCPMBytePairTokenizer(
        tokenizerJSONURL: localMiniCPMTokenizerJSONURL(),
        addBosToken: true
    )
    let promptSuiteURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        .appending(path: "tools/benchmark/fixtures/benchmark-prompts.json")
    let suite = try RuntimeBenchmarkPromptSuite.load(from: promptSuiteURL)
    let expectedTokenIDsByPromptID: [String: [Int32]] = [
        "zh-short-001": [0, 4863, 1066, 2003, 15008, 12778, 10744, 119516, 7192, 45921, 59048, 4290, 18487, 45105, 43449, 396],
        "zh-short-002": [0, 2418, 41058, 26005, 3890, 5351, 61309, 6373, 1039, 18884, 5817, 1585, 9630, 18086, 45025, 11302, 396],
        "en-short-001": [0, 77524, 310, 678, 2871, 14504, 3212, 280, 11420, 826, 4871, 54418, 11454, 4245, 10021, 5297, 19292, 35],
        "en-short-002": [0, 18579, 4709, 1144, 285, 92892, 1118, 1180, 41, 438, 24604, 12537, 316, 1180, 45, 4418, 12537, 374, 280, 242, 38, 55, 1327, 35],
        "code-fix-001": [0, 58107, 533, 42709, 1323, 47, 1323, 48057, 4184, 33, 1432, 33, 3240, 30, 319, 2021, 7479, 9337, 88032, 33, 7479, 9552, 67682, 33, 769, 12271, 324],
        "code-fix-002": [0, 58107, 533, 41625, 4435, 702, 411, 39432, 285, 6187, 1886, 47, 1805, 10769, 483, 3240, 29, 6036, 9118, 33, 8091, 30],
        "watch-utility-001": [0, 62, 558, 242, 627, 4533, 1482, 280, 1455, 35, 46882, 451, 280, 4052, 5063, 60639, 354, 571, 1655, 376, 280, 5297, 35],
        "watch-utility-002": [0, 28897, 533, 844, 280, 44375, 5297, 46132, 47, 416, 1327, 18629, 8626, 51302, 316, 357, 6066, 374, 49038, 1170, 35],
        "safety-refusal-001": [0, 38929, 559, 3848, 16482, 23276, 12613, 317, 32039, 4517, 3342, 468, 13311, 40456, 1397, 35],
        "safety-refusal-002": [0, 3975, 571, 354, 31502, 285, 800, 7833, 4368, 5856, 374, 280, 10550, 1327, 25550, 52]
    ]

    for prompt in suite.prompts {
        #expect(try tokenizer.encode(prompt.input) == expectedTokenIDsByPromptID[prompt.id])
    }
}
#endif

@Test func miniCPMChatTemplateRendersNoThinkFastPath() throws {
    let template = MiniCPMChatTemplate(bosToken: "<bos>")
    let rendered = template.render(
        messages: [
            ChatMessage(role: .system, content: "You run locally on Apple Watch."),
            ChatMessage(role: .user, content: "Answer briefly.")
        ],
        addGenerationPrompt: true,
        enableThinking: false
    )

    let expected = "<bos><|im_start|>system\n" +
        "You run locally on Apple Watch.<|im_end|>\n" +
        "<|im_start|>user\n" +
        "Answer briefly.<|im_end|>\n" +
        "<|im_start|>assistant\n" +
        "<think>\n\n</think>\n\n"
    #expect(rendered == expected)
}

@Test func qwen3ChatTemplateRendersNoThinkFastPath() throws {
    let template = Qwen3ChatTemplate()
    let rendered = template.render(
        messages: [ChatMessage(role: .user, content: "Answer briefly. What is 2+2?")],
        addGenerationPrompt: true,
        enableThinking: false
    )

    let expected = "<|im_start|>user\n" +
        "Answer briefly. What is 2+2?<|im_end|>\n" +
        "<|im_start|>assistant\n" +
        "<think>\n\n</think>\n\n"
    #expect(rendered == expected)
}

#if os(macOS)
@Test func qwen3BytePairTokenizerMatchesHFNoThinkTemplateSmoke() throws {
    let tokenizer = try MiniCPMBytePairTokenizer(
        tokenizerJSONURL: localQwen3TokenizerJSONURL(),
        addBosToken: false,
        eosTokenIDs: [151645]
    )
    let rendered = Qwen3ChatTemplate().render(
        messages: [ChatMessage(role: .user, content: "Answer briefly. What is 2+2?")],
        addGenerationPrompt: true,
        enableThinking: false
    )

    #expect(try tokenizer.encode(rendered) == [
        151644, 872, 198, 16141, 26753, 13, 3555, 374,
        220, 17, 10, 17, 30, 151645, 198, 151644, 77091,
        198, 151667, 271, 151668, 271
    ])
}

@Test func qwen3BytePairTokenizerExposesDecodableTokenUpperBound() throws {
    let tokenizer = try MiniCPMBytePairTokenizer(
        tokenizerJSONURL: localQwen3TokenizerJSONURL(),
        addBosToken: false,
        eosTokenIDs: [151645]
    )

    #expect(tokenizer.decodableTokenIDUpperBound == 151669)
    #expect(try tokenizer.decode(tokenIDs: [151668]) == "</think>")
    #expect(throws: MiniCPMTokenizerError.unknownTokenID(151680)) {
        _ = try tokenizer.decode(tokenIDs: [151680])
    }
}
#endif

#if os(macOS)
private func localMiniCPMTokenizerJSONURL() -> URL {
    URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        .appending(path: "artifacts/hf/MiniCPM5-1B/tokenizer.json")
}

private func localQwen3TokenizerJSONURL() -> URL {
    URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        .appending(path: "artifacts/hf/Qwen3-0.6B/tokenizer.json")
}
#endif

@Test func miniCPMSpecialTokensMatchPublishedConfig() {
    #expect(MiniCPMSpecialTokens.bosTokenID == 0)
    #expect(MiniCPMSpecialTokens.padTokenID == 1)
    #expect(MiniCPMSpecialTokens.eosTokenIDs == [1, 130073])
    #expect(MiniCPMSpecialTokens.vocabularySize == 130560)
}

@Test func miniCPMInt8KVCacheBudgetMatchesArchitecture() {
    let descriptor = KVCacheDescriptor.miniCPM5(contextTokens: 512, precision: .int8)

    #expect(descriptor.layers == 24)
    #expect(descriptor.kvHeads == 2)
    #expect(descriptor.headDimension == 128)
    #expect(descriptor.bytesPerToken == 12_288)
    #expect(descriptor.totalBytes == 6_291_456)
}
