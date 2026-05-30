import Foundation
import Testing
@testable import WatchLMCore

@Test func quantizationCalibrationSuiteLoadsWatchPromptFixture() throws {
    let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let fixtureURL = root.appending(path: "tools/benchmark/fixtures/calibration-prompts.json")

    let suite = try QuantizationCalibrationSuite.load(from: fixtureURL)

    #expect(suite.modelID == "openbmb/MiniCPM5-1B")
    #expect(suite.tokenizerSource == "openbmb/MiniCPM5-1B")
    #expect(suite.contextTokens == 256)
    #expect(suite.promptFormat == "minicpm5-chat-template-no-think")
    #expect(suite.prefixTokenCounts == [1, 2, 4, 8, 12, 18, 32])
    #expect(suite.prompts.count == 12)
    #expect(suite.validationErrors.isEmpty)

    let categories = Set(suite.prompts.map(\.category))
    #expect(categories == Set(QuantizationCalibrationSuite.requiredCategories))
    #expect(suite.prompts.allSatisfy { prompt in
        prompt.renderedPrompt.contains("<think>\n\n</think>\n\n")
    })

    let benchmarkPrompts = suite.benchmarkPrompts(maxNewTokens: 2)
    #expect(benchmarkPrompts.count == suite.prompts.count)
    #expect(benchmarkPrompts[0].id == suite.prompts[0].id)
    #expect(benchmarkPrompts[0].input == suite.prompts[0].renderedPrompt)
    #expect(benchmarkPrompts[0].maxNewTokens == 2)
}

@Test func quantizationCalibrationSuiteReportsValidationErrorsTogether() {
    let suite = QuantizationCalibrationSuite(
        schemaVersion: 2,
        modelID: "other/model",
        tokenizerSource: "",
        contextTokens: 128,
        promptFormat: "raw",
        prefixTokenCounts: [4, 2, 512],
        prompts: [
            QuantizationCalibrationPrompt(
                id: "",
                category: "unknown",
                language: "",
                messages: [],
                renderedPrompt: "hello",
                maxNewTokens: 0,
                tags: []
            )
        ]
    )

    let errors = suite.validationErrors.joined(separator: "\n")
    #expect(errors.contains("schemaVersion must be 1"))
    #expect(errors.contains("modelId must be openbmb/MiniCPM5-1B"))
    #expect(errors.contains("tokenizerSource must be openbmb/MiniCPM5-1B"))
    #expect(errors.contains("contextTokens must be 256"))
    #expect(errors.contains("promptFormat must be minicpm5-chat-template-no-think"))
    #expect(errors.contains("prefixTokenCounts must be strictly increasing"))
    #expect(errors.contains("prefixTokenCounts must be <= contextTokens"))
    #expect(errors.contains("prompt[0].id must be a non-empty string"))
    #expect(errors.contains("prompt[0].category is unsupported"))
    #expect(errors.contains("prompt[0].language must be a non-empty string"))
    #expect(errors.contains("prompt[0].messages must be a non-empty array"))
    #expect(errors.contains("prompt[0].renderedPrompt must use the MiniCPM no-think assistant prefix"))
    #expect(errors.contains("prompt[0].maxNewTokens must be between 1 and 96"))
    #expect(errors.contains("prompt[0].tags must be a non-empty array"))
    #expect(errors.contains("missing required category zh_short_instruction"))
}
