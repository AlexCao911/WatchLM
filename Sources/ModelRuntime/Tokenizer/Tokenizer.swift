import Foundation

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

public enum MiniCPMTokenizerError: Error, Equatable, Sendable {
    case invalidTokenizerJSON(String)
    case missingToken(String)
    case unknownTokenID(Int32)
}

public struct MiniCPMBytePairTokenizer: TextTokenizer {
    public let endOfSequenceTokenIDs: Set<Int32>

    private let addBosToken: Bool
    private let bosTokenID: Int32
    private let vocab: [String: Int32]
    private let tokenByID: [Int32: String]
    private let bpeRanks: [String: Int]
    private let addedTokenByContent: [String: Int32]
    private let addedTokenContentByID: [Int32: String]
    private let addedTokenContentsByLength: [String]
    private let byteEncoder: [UInt8: String]
    private let byteDecoder: [UnicodeScalar: UInt8]
    private let unknownTokenID: Int32?

    public init(
        tokenizerJSONURL: URL,
        addBosToken: Bool = true,
        bosTokenID: Int32 = MiniCPMSpecialTokens.bosTokenID,
        eosTokenIDs: Set<Int32> = MiniCPMSpecialTokens.eosTokenIDs
    ) throws {
        let data = try Data(contentsOf: tokenizerJSONURL)
        try self.init(
            tokenizerJSONData: data,
            addBosToken: addBosToken,
            bosTokenID: bosTokenID,
            eosTokenIDs: eosTokenIDs
        )
    }

    public init(
        tokenizerJSONData data: Data,
        addBosToken: Bool = true,
        bosTokenID: Int32 = MiniCPMSpecialTokens.bosTokenID,
        eosTokenIDs: Set<Int32> = MiniCPMSpecialTokens.eosTokenIDs
    ) throws {
        let root = try Self.dictionary(from: data)
        guard let model = root["model"] as? [String: Any] else {
            throw MiniCPMTokenizerError.invalidTokenizerJSON("tokenizer.json must include a model object.")
        }

        guard let rawVocab = model["vocab"] as? [String: Any] else {
            throw MiniCPMTokenizerError.invalidTokenizerJSON("tokenizer model must include vocab.")
        }

        var vocab: [String: Int32] = [:]
        vocab.reserveCapacity(rawVocab.count)
        for (token, rawID) in rawVocab {
            guard let id = Self.int32(from: rawID) else {
                throw MiniCPMTokenizerError.invalidTokenizerJSON("vocab entry \(token) must have an integer id.")
            }
            vocab[token] = id
        }

        var bpeRanks: [String: Int] = [:]
        if let rawMerges = model["merges"] as? [[String]] {
            for (rank, pair) in rawMerges.enumerated() where pair.count == 2 {
                bpeRanks[Self.pairKey(pair[0], pair[1])] = rank
            }
        } else if let rawMerges = model["merges"] as? [String] {
            for (rank, merge) in rawMerges.enumerated() {
                let pair = merge.split(separator: " ", maxSplits: 1).map(String.init)
                if pair.count == 2 {
                    bpeRanks[Self.pairKey(pair[0], pair[1])] = rank
                }
            }
        } else {
            throw MiniCPMTokenizerError.invalidTokenizerJSON("tokenizer model must include BPE merges.")
        }

        var addedTokenByContent: [String: Int32] = [:]
        var addedTokenContentByID: [Int32: String] = [:]
        if let rawAddedTokens = root["added_tokens"] as? [[String: Any]] {
            for rawToken in rawAddedTokens {
                guard let content = rawToken["content"] as? String,
                      let id = Self.int32(from: rawToken["id"])
                else {
                    continue
                }
                addedTokenByContent[content] = id
                addedTokenContentByID[id] = content
            }
        }

        let byteCodec = Self.makeByteLevelCodec()
        self.addBosToken = addBosToken
        self.bosTokenID = bosTokenID
        endOfSequenceTokenIDs = eosTokenIDs
        self.vocab = vocab
        tokenByID = Dictionary(uniqueKeysWithValues: vocab.map { ($0.value, $0.key) })
        self.bpeRanks = bpeRanks
        self.addedTokenByContent = addedTokenByContent
        self.addedTokenContentByID = addedTokenContentByID
        addedTokenContentsByLength = addedTokenByContent.keys.sorted {
            if $0.count == $1.count {
                return $0 < $1
            }
            return $0.count > $1.count
        }
        byteEncoder = byteCodec.encoder
        byteDecoder = byteCodec.decoder
        unknownTokenID = addedTokenByContent["<unk>"] ?? vocab["<unk>"]
    }

    public func encode(_ text: String) throws -> [Int32] {
        var tokenIDs: [Int32] = []
        if addBosToken {
            tokenIDs.append(bosTokenID)
        }

        for piece in splitAddedTokens(in: text) {
            switch piece {
            case .addedToken(let id):
                tokenIDs.append(id)
            case .text(let text):
                for pretoken in try pretokenize(text) where !pretoken.isEmpty {
                    let byteToken = encodeBytes(pretoken)
                    for bpeToken in bpe(byteToken) {
                        if let id = vocab[bpeToken] {
                            tokenIDs.append(id)
                        } else if let unknownTokenID {
                            tokenIDs.append(unknownTokenID)
                        } else {
                            throw MiniCPMTokenizerError.missingToken(bpeToken)
                        }
                    }
                }
            }
        }

        return tokenIDs
    }

    public func decode(tokenIDs: [Int32]) throws -> String {
        var output = ""
        var pendingBytes: [UInt8] = []

        func flushBytes() {
            guard !pendingBytes.isEmpty else {
                return
            }
            output += String(decoding: pendingBytes, as: UTF8.self)
            pendingBytes.removeAll(keepingCapacity: true)
        }

        for tokenID in tokenIDs {
            if let addedToken = addedTokenContentByID[tokenID] {
                flushBytes()
                output += addedToken
                continue
            }

            guard let token = tokenByID[tokenID] else {
                throw MiniCPMTokenizerError.unknownTokenID(tokenID)
            }

            for scalar in token.unicodeScalars {
                if let byte = byteDecoder[scalar] {
                    pendingBytes.append(byte)
                } else {
                    flushBytes()
                    output += String(scalar)
                }
            }
        }

        flushBytes()
        return output
    }

    private func splitAddedTokens(in text: String) -> [TokenPiece] {
        guard !addedTokenContentsByLength.isEmpty else {
            return [.text(text)]
        }

        var pieces: [TokenPiece] = []
        var bufferedText = ""
        var index = text.startIndex

        while index < text.endIndex {
            let suffix = text[index...]
            if let content = addedTokenContentsByLength.first(where: { suffix.hasPrefix($0) }),
               let tokenID = addedTokenByContent[content] {
                if !bufferedText.isEmpty {
                    pieces.append(.text(bufferedText))
                    bufferedText.removeAll(keepingCapacity: true)
                }
                pieces.append(.addedToken(tokenID))
                index = text.index(index, offsetBy: content.count)
            } else {
                bufferedText.append(text[index])
                index = text.index(after: index)
            }
        }

        if !bufferedText.isEmpty {
            pieces.append(.text(bufferedText))
        }

        return pieces
    }

    private func pretokenize(_ text: String) throws -> [String] {
        let digitRegex = try NSRegularExpression(pattern: "\\p{N}{1,3}")
        let tokenRegex = try NSRegularExpression(
            pattern: "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}+| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
        )
        return Self.splitIsolated(Self.splitIsolated([text], using: digitRegex), using: tokenRegex)
    }

    private func encodeBytes(_ text: String) -> String {
        text.utf8.map { byteEncoder[$0] ?? "" }.joined()
    }

    private func bpe(_ token: String) -> [String] {
        var word = token.unicodeScalars.map { String($0) }
        guard word.count > 1 else {
            return word
        }

        while true {
            var bestRank: Int?
            var bestPair: (String, String)?
            for index in 0..<(word.count - 1) {
                let pair = (word[index], word[index + 1])
                guard let rank = bpeRanks[Self.pairKey(pair.0, pair.1)] else {
                    continue
                }

                if bestRank == nil || rank < bestRank! {
                    bestRank = rank
                    bestPair = pair
                }
            }

            guard let bestPair else {
                break
            }

            var merged: [String] = []
            var index = 0
            while index < word.count {
                if index < word.count - 1 && word[index] == bestPair.0 && word[index + 1] == bestPair.1 {
                    merged.append(bestPair.0 + bestPair.1)
                    index += 2
                } else {
                    merged.append(word[index])
                    index += 1
                }
            }
            word = merged

            if word.count == 1 {
                break
            }
        }

        return word
    }

    private static func splitIsolated(_ inputs: [String], using regex: NSRegularExpression) -> [String] {
        inputs.flatMap { text in
            splitIsolated(text, using: regex)
        }
    }

    private static func splitIsolated(_ text: String, using regex: NSRegularExpression) -> [String] {
        let matches = regex.matches(
            in: text,
            range: NSRange(text.startIndex..<text.endIndex, in: text)
        )
        guard !matches.isEmpty else {
            return text.isEmpty ? [] : [text]
        }

        var pieces: [String] = []
        var cursor = text.startIndex
        for match in matches {
            guard let range = Range(match.range, in: text) else {
                continue
            }

            if cursor < range.lowerBound {
                pieces.append(String(text[cursor..<range.lowerBound]))
            }
            pieces.append(String(text[range]))
            cursor = range.upperBound
        }

        if cursor < text.endIndex {
            pieces.append(String(text[cursor..<text.endIndex]))
        }

        return pieces.filter { !$0.isEmpty }
    }

    private static func makeByteLevelCodec() -> (
        encoder: [UInt8: String],
        decoder: [UnicodeScalar: UInt8]
    ) {
        var bytes = Array(33...126).map(UInt8.init)
        bytes += Array(161...172).map(UInt8.init)
        bytes += Array(174...255).map(UInt8.init)

        var codePoints = bytes.map(Int.init)
        var nextCodePoint = 0
        for byteValue in 0...255 {
            let byte = UInt8(byteValue)
            if !bytes.contains(byte) {
                bytes.append(byte)
                codePoints.append(256 + nextCodePoint)
                nextCodePoint += 1
            }
        }

        var encoder: [UInt8: String] = [:]
        var decoder: [UnicodeScalar: UInt8] = [:]
        for (byte, codePoint) in zip(bytes, codePoints) {
            guard let scalar = UnicodeScalar(codePoint) else {
                continue
            }
            encoder[byte] = String(scalar)
            decoder[scalar] = byte
        }
        return (encoder, decoder)
    }

    private static func dictionary(from data: Data) throws -> [String: Any] {
        guard let dictionary = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw MiniCPMTokenizerError.invalidTokenizerJSON("tokenizer.json root must be an object.")
        }
        return dictionary
    }

    private static func int32(from value: Any?) -> Int32? {
        if let number = value as? NSNumber {
            return number.int32Value
        }
        if let integer = value as? Int {
            return Int32(integer)
        }
        return nil
    }

    private static func pairKey(_ left: String, _ right: String) -> String {
        left + "\u{0001}" + right
    }

    private enum TokenPiece {
        case text(String)
        case addedToken(Int32)
    }
}
