import Darwin
import Foundation
import WatchLMBenchmarkSupport

@main
struct WatchLMBenchmarkMain {
    static func main() async {
        do {
            let options = try RuntimeBenchmarkCommandOptions.parse(Array(CommandLine.arguments.dropFirst()))
            let command = RuntimeBenchmarkCommand(options: options)

            if options.stagingPlanOnly {
                let plan = try command.runStagingPlan()
                if let outputURL = options.outputURL {
                    print("wrote model asset staging plan: \(outputURL.path)")
                    print(
                        "items: \(plan.items.count), " +
                        "total_bytes: \(plan.totalByteCount), " +
                        "destination: \(plan.destinationRootDescription)"
                    )
                } else {
                    let data = try RuntimeBenchmarkCommand.encode(stagingPlan: plan)
                    FileHandle.standardOutput.write(data)
                    FileHandle.standardOutput.write(Data("\n".utf8))
                }
            } else if options.runsSensitivityComparison {
                let report = try command.runSensitivityComparison()
                if let outputURL = options.outputURL {
                    print("wrote quantization sensitivity report: \(outputURL.path)")
                    print(
                        "gate_ok: \(report.gate.ok), " +
                        "avg_prefill_topk_overlap: \(report.summary.averagePrefillTopKOverlapRatio)"
                    )
                } else {
                    let data = try RuntimeBenchmarkCommand.encode(sensitivityReport: report)
                    FileHandle.standardOutput.write(data)
                    FileHandle.standardOutput.write(Data("\n".utf8))
                }
            } else if options.diagnosticsTopK != nil {
                let report = try command.runDiagnostics()
                if let outputURL = options.outputURL {
                    print("wrote Core ML diagnostics report: \(outputURL.path)")
                    print("prompts: \(report.summary.succeededPromptCount)/\(report.summary.promptCount), top_k: \(report.topK)")
                } else {
                    let data = try RuntimeBenchmarkCommand.encode(diagnosticsReport: report)
                    FileHandle.standardOutput.write(data)
                    FileHandle.standardOutput.write(Data("\n".utf8))
                }
            } else {
                let report = try await command.run()

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
            }
        } catch {
            FileHandle.standardError.write(Data("error: \(error)\n".utf8))
            exit(1)
        }
    }
}
