import Foundation

public struct QuantizationCalibrationPrompt: Codable, Equatable, Sendable {
    public var id: String
    public var category: String
    public var language: String
    public var messages: [ChatMessage]
    public var renderedPrompt: String
    public var maxNewTokens: Int
    public var tags: [String]

    public init(
        id: String,
        category: String,
        language: String,
        messages: [ChatMessage],
        renderedPrompt: String,
        maxNewTokens: Int,
        tags: [String]
    ) {
        self.id = id
        self.category = category
        self.language = language
        self.messages = messages
        self.renderedPrompt = renderedPrompt
        self.maxNewTokens = maxNewTokens
        self.tags = tags
    }
}

public enum QuantizationCalibrationSuiteError: Error, Equatable, Sendable {
    case invalidSuite([String])
}

public struct QuantizationCalibrationSuite: Codable, Equatable, Sendable {
    public static let requiredCategories = [
        "zh_short_instruction",
        "en_short_instruction",
        "watch_utility",
        "code_small_fix",
        "stop_sequence",
        "safety_refusal",
    ]

    private static let supportedSchemaVersion = 1
    private static let expectedModelID = "openbmb/MiniCPM5-1B"
    private static let expectedPromptFormat = "minicpm5-chat-template-no-think"
    private static let expectedContextTokens = 256
    private static let minNewTokens = 1
    private static let maxNewTokens = 96
    private static let noThinkAssistantPrefix = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    public var schemaVersion: Int
    public var modelID: String
    public var tokenizerSource: String
    public var contextTokens: Int
    public var promptFormat: String
    public var prefixTokenCounts: [Int]
    public var prompts: [QuantizationCalibrationPrompt]

    public init(
        schemaVersion: Int,
        modelID: String,
        tokenizerSource: String,
        contextTokens: Int,
        promptFormat: String,
        prefixTokenCounts: [Int],
        prompts: [QuantizationCalibrationPrompt]
    ) {
        self.schemaVersion = schemaVersion
        self.modelID = modelID
        self.tokenizerSource = tokenizerSource
        self.contextTokens = contextTokens
        self.promptFormat = promptFormat
        self.prefixTokenCounts = prefixTokenCounts
        self.prompts = prompts
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion
        case modelID = "modelId"
        case tokenizerSource
        case contextTokens
        case promptFormat
        case prefixTokenCounts
        case prompts
    }

    public static func load(from url: URL) throws -> QuantizationCalibrationSuite {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(QuantizationCalibrationSuite.self, from: data)
    }

    public var validationErrors: [String] {
        var errors: [String] = []

        if schemaVersion != Self.supportedSchemaVersion {
            errors.append("schemaVersion must be \(Self.supportedSchemaVersion)")
        }

        if modelID != Self.expectedModelID {
            errors.append("modelId must be \(Self.expectedModelID)")
        }

        if tokenizerSource != Self.expectedModelID {
            errors.append("tokenizerSource must be \(Self.expectedModelID)")
        }

        if contextTokens != Self.expectedContextTokens {
            errors.append("contextTokens must be \(Self.expectedContextTokens)")
        }

        if promptFormat != Self.expectedPromptFormat {
            errors.append("promptFormat must be \(Self.expectedPromptFormat)")
        }

        validatePrefixTokenCounts(into: &errors)
        validatePrompts(into: &errors)

        return errors
    }

    public func validated() throws -> QuantizationCalibrationSuite {
        let errors = validationErrors
        guard errors.isEmpty else {
            throw QuantizationCalibrationSuiteError.invalidSuite(errors)
        }
        return self
    }

    public func benchmarkPrompts(maxNewTokens overrideMaxNewTokens: Int? = nil) -> [RuntimeBenchmarkPrompt] {
        prompts.map { prompt in
            RuntimeBenchmarkPrompt(
                id: prompt.id,
                category: prompt.category,
                language: prompt.language,
                input: prompt.renderedPrompt,
                maxNewTokens: overrideMaxNewTokens ?? prompt.maxNewTokens,
                qualityChecks: prompt.tags
            )
        }
    }

    private func validatePrefixTokenCounts(into errors: inout [String]) {
        if prefixTokenCounts.isEmpty {
            errors.append("prefixTokenCounts must be a non-empty array")
            return
        }

        if prefixTokenCounts.contains(where: { $0 <= 0 }) {
            errors.append("prefixTokenCounts must contain positive integers")
        }

        if zip(prefixTokenCounts, prefixTokenCounts.dropFirst()).contains(where: >=) {
            errors.append("prefixTokenCounts must be strictly increasing")
        }

        if prefixTokenCounts.contains(where: { $0 > contextTokens }) {
            errors.append("prefixTokenCounts must be <= contextTokens")
        }
    }

    private func validatePrompts(into errors: inout [String]) {
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

            if Self.requiredCategories.contains(prompt.category) {
                seenCategories.insert(prompt.category)
            } else {
                errors.append("prompt[\(index)].category is unsupported")
            }

            if prompt.language.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                errors.append("prompt[\(index)].language must be a non-empty string")
            }

            if prompt.messages.isEmpty {
                errors.append("prompt[\(index)].messages must be a non-empty array")
            } else if prompt.messages.contains(where: { $0.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }) {
                errors.append("prompt[\(index)].messages must not contain empty content")
            }

            if !usesMiniCPMNoThinkTemplate(prompt.renderedPrompt) {
                errors.append("prompt[\(index)].renderedPrompt must use the MiniCPM no-think assistant prefix")
            }

            if prompt.maxNewTokens < Self.minNewTokens || prompt.maxNewTokens > Self.maxNewTokens {
                errors.append("prompt[\(index)].maxNewTokens must be between \(Self.minNewTokens) and \(Self.maxNewTokens)")
            }

            if prompt.tags.isEmpty || prompt.tags.contains(where: { $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }) {
                errors.append("prompt[\(index)].tags must be a non-empty array")
            }
        }

        for category in Self.requiredCategories where !seenCategories.contains(category) {
            errors.append("missing required category \(category)")
        }
    }

    private func usesMiniCPMNoThinkTemplate(_ renderedPrompt: String) -> Bool {
        renderedPrompt.hasPrefix("<s><|im_start|>system\n") &&
            renderedPrompt.contains(Self.noThinkAssistantPrefix)
    }
}
