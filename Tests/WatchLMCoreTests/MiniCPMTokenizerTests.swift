import Testing
@testable import WatchLMCore

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
