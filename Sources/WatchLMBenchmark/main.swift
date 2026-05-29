import Darwin
import Foundation
import WatchLMBenchmarkSupport

@main
struct WatchLMBenchmarkMain {
    static func main() async {
        do {
            let options = try RuntimeBenchmarkCommandOptions.parse(Array(CommandLine.arguments.dropFirst()))
            let report = try await RuntimeBenchmarkCommand(options: options).run()

            if let outputURL = options.outputURL {
                let tokenAgreement = report.summary.averageTokenAgreement.map { String($0) } ?? "n/a"
                print("wrote benchmark report: \(outputURL.path)")
                print(
                    "prompts: \(report.summary.succeededPromptCount)/\(report.summary.promptCount), " +
                    "avg_token_agreement: \(tokenAgreement)"
                )
            } else {
                let data = try RuntimeBenchmarkCommand.encode(report: report)
                FileHandle.standardOutput.write(data)
                FileHandle.standardOutput.write(Data("\n".utf8))
            }
        } catch {
            FileHandle.standardError.write(Data("error: \(error)\n".utf8))
            exit(1)
        }
    }
}
