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

#if os(macOS)
private func localMiniCPMTokenizerJSONURL() -> URL {
    URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        .appending(path: "artifacts/hf/MiniCPM5-1B/tokenizer.json")
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
