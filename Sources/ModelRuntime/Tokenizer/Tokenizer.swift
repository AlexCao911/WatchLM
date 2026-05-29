public struct ChatMessage: Codable, Equatable, Sendable {
    public enum Role: String, Codable, Sendable {
        case system
        case user
        case assistant
        case tool
    }

    public var role: Role
    public var content: String

    public init(role: Role, content: String) {
        self.role = role
        self.content = content
    }
}

public protocol TextTokenizer: Sendable {
    var endOfSequenceTokenIDs: Set<Int32> { get }

    func encode(_ text: String) throws -> [Int32]

    func decode(tokenIDs: [Int32]) throws -> String
}

public enum MiniCPMSpecialTokens {
    public static let bosTokenID: Int32 = 0
    public static let padTokenID: Int32 = 1
    public static let eosTokenIDs: Set<Int32> = [1, 130073]
    public static let vocabularySize = 130560
}

public struct MiniCPMChatTemplate: Sendable {
    private let bosToken: String

    public init(bosToken: String) {
        self.bosToken = bosToken
    }

    public func render(
        messages: [ChatMessage],
        addGenerationPrompt: Bool,
        enableThinking: Bool
    ) -> String {
        var rendered = bosToken

        for message in messages {
            rendered += "<|im_start|>\(message.role.rawValue)\n"
            rendered += message.content
            rendered += "<|im_end|>\n"
        }

        if addGenerationPrompt {
            rendered += "<|im_start|>assistant\n"
            if enableThinking {
                rendered += "<think>\n"
            } else {
                rendered += "<think>\n\n</think>\n\n"
            }
        }

        return rendered
    }
}
