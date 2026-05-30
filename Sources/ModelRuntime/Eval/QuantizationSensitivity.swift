import Foundation

public struct LogitsDiagnosticPoint: Codable, Equatable, Sendable {
    public var promptID: String
    public var category: String
    public var language: String
    public var prefixTokenCount: Int
    public var prefillTopK: [TokenLogit]
    public var decodeTopK: [TokenLogit]

    public init(
        promptID: String,
        category: String,
        language: String,
        prefixTokenCount: Int,
        prefillTopK: [TokenLogit],
        decodeTopK: [TokenLogit] = []
    ) {
        self.promptID = promptID
        self.category = category
        self.language = language
        self.prefixTokenCount = prefixTokenCount
        self.prefillTopK = prefillTopK
        self.decodeTopK = decodeTopK
    }
}

public struct QuantizationSensitivityTargets: Codable, Equatable, Sendable {
    public var minimumAveragePrefillTopKOverlapRatio: Double
    public var criticalPrefixTokenCount: Int
    public var minimumCriticalPrefixOverlapCount: Int

    public init(
        minimumAveragePrefillTopKOverlapRatio: Double = 0.8,
        criticalPrefixTokenCount: Int = 4,
        minimumCriticalPrefixOverlapCount: Int = 1
    ) {
        self.minimumAveragePrefillTopKOverlapRatio = minimumAveragePrefillTopKOverlapRatio
        self.criticalPrefixTokenCount = criticalPrefixTokenCount
        self.minimumCriticalPrefixOverlapCount = minimumCriticalPrefixOverlapCount
    }
}

public struct QuantizationSensitivityComparison: Codable, Equatable, Sendable {
    public var promptID: String
    public var category: String
    public var language: String
    public var prefixTokenCount: Int
    public var baselinePrefillTop1TokenID: Int32?
    public var candidatePrefillTop1TokenID: Int32?
    public var prefillTop1Matches: Bool
    public var prefillTopKOverlapCount: Int
    public var prefillTopKOverlapRatio: Double
    public var decodeTopKOverlapCount: Int?
    public var decodeTopKOverlapRatio: Double?

    public init(
        promptID: String,
        category: String,
        language: String,
        prefixTokenCount: Int,
        baselinePrefillTop1TokenID: Int32?,
        candidatePrefillTop1TokenID: Int32?,
        prefillTop1Matches: Bool,
        prefillTopKOverlapCount: Int,
        prefillTopKOverlapRatio: Double,
        decodeTopKOverlapCount: Int? = nil,
        decodeTopKOverlapRatio: Double? = nil
    ) {
        self.promptID = promptID
        self.category = category
        self.language = language
        self.prefixTokenCount = prefixTokenCount
        self.baselinePrefillTop1TokenID = baselinePrefillTop1TokenID
        self.candidatePrefillTop1TokenID = candidatePrefillTop1TokenID
        self.prefillTop1Matches = prefillTop1Matches
        self.prefillTopKOverlapCount = prefillTopKOverlapCount
        self.prefillTopKOverlapRatio = prefillTopKOverlapRatio
        self.decodeTopKOverlapCount = decodeTopKOverlapCount
        self.decodeTopKOverlapRatio = decodeTopKOverlapRatio
    }
}

public struct QuantizationSensitivitySummary: Codable, Equatable, Sendable {
    public var baselinePointCount: Int
    public var candidatePointCount: Int
    public var comparedPointCount: Int
    public var averagePrefillTopKOverlapRatio: Double
    public var prefillTop1Agreement: Double
    public var firstZeroPrefillOverlapPrefixTokenCount: Int?

    public init(
        baselinePointCount: Int,
        candidatePointCount: Int,
        comparedPointCount: Int,
        averagePrefillTopKOverlapRatio: Double,
        prefillTop1Agreement: Double,
        firstZeroPrefillOverlapPrefixTokenCount: Int? = nil
    ) {
        self.baselinePointCount = baselinePointCount
        self.candidatePointCount = candidatePointCount
        self.comparedPointCount = comparedPointCount
        self.averagePrefillTopKOverlapRatio = averagePrefillTopKOverlapRatio
        self.prefillTop1Agreement = prefillTop1Agreement
        self.firstZeroPrefillOverlapPrefixTokenCount = firstZeroPrefillOverlapPrefixTokenCount
    }
}

public struct QuantizationSensitivityGate: Codable, Equatable, Sendable {
    public var ok: Bool
    public var failures: [String]
    public var targets: QuantizationSensitivityTargets

    public init(
        ok: Bool,
        failures: [String],
        targets: QuantizationSensitivityTargets
    ) {
        self.ok = ok
        self.failures = failures
        self.targets = targets
    }
}

public struct QuantizationSensitivityReport: Codable, Equatable, Sendable {
    public var baselinePolicyID: String
    public var candidatePolicyID: String
    public var comparisons: [QuantizationSensitivityComparison]
    public var summary: QuantizationSensitivitySummary
    public var gate: QuantizationSensitivityGate

    public init(
        baselinePolicyID: String,
        candidatePolicyID: String,
        comparisons: [QuantizationSensitivityComparison],
        summary: QuantizationSensitivitySummary,
        gate: QuantizationSensitivityGate
    ) {
        self.baselinePolicyID = baselinePolicyID
        self.candidatePolicyID = candidatePolicyID
        self.comparisons = comparisons
        self.summary = summary
        self.gate = gate
    }
}

public enum QuantizationSensitivityError: Error, Equatable, Sendable {
    case duplicatePoint(String)
    case missingCandidatePoint(String)
}

public enum QuantizationSensitivityScorer {
    public static func compare(
        baselinePolicyID: String,
        candidatePolicyID: String,
        baseline: [LogitsDiagnosticPoint],
        candidate: [LogitsDiagnosticPoint],
        targets: QuantizationSensitivityTargets = QuantizationSensitivityTargets()
    ) throws -> QuantizationSensitivityReport {
        let candidateByKey = try keyedPoints(candidate)
        let comparisons = try baseline.map { baselinePoint in
            let key = diagnosticKey(for: baselinePoint)
            guard let candidatePoint = candidateByKey[key] else {
                throw QuantizationSensitivityError.missingCandidatePoint(key)
            }
            return comparison(baseline: baselinePoint, candidate: candidatePoint)
        }.sorted {
            if $0.promptID == $1.promptID {
                return $0.prefixTokenCount < $1.prefixTokenCount
            }
            return $0.promptID < $1.promptID
        }

        _ = try keyedPoints(baseline)
        let summary = QuantizationSensitivitySummary(comparisons: comparisons, baseline: baseline, candidate: candidate)
        let gate = QuantizationSensitivityGate(summary: summary, comparisons: comparisons, targets: targets)
        return QuantizationSensitivityReport(
            baselinePolicyID: baselinePolicyID,
            candidatePolicyID: candidatePolicyID,
            comparisons: comparisons,
            summary: summary,
            gate: gate
        )
    }

    private static func keyedPoints(_ points: [LogitsDiagnosticPoint]) throws -> [String: LogitsDiagnosticPoint] {
        var keyed: [String: LogitsDiagnosticPoint] = [:]
        for point in points {
            let key = diagnosticKey(for: point)
            guard keyed[key] == nil else {
                throw QuantizationSensitivityError.duplicatePoint(key)
            }
            keyed[key] = point
        }
        return keyed
    }

    private static func comparison(
        baseline: LogitsDiagnosticPoint,
        candidate: LogitsDiagnosticPoint
    ) -> QuantizationSensitivityComparison {
        let prefillOverlap = overlapCount(
            baseline.prefillTopK.map(\.tokenID),
            candidate.prefillTopK.map(\.tokenID)
        )
        let decodeOverlap = decodeOverlapComparison(baseline: baseline, candidate: candidate)
        let baselineTop1 = baseline.prefillTopK.first?.tokenID
        let candidateTop1 = candidate.prefillTopK.first?.tokenID
        return QuantizationSensitivityComparison(
            promptID: baseline.promptID,
            category: baseline.category,
            language: baseline.language,
            prefixTokenCount: baseline.prefixTokenCount,
            baselinePrefillTop1TokenID: baselineTop1,
            candidatePrefillTop1TokenID: candidateTop1,
            prefillTop1Matches: baselineTop1 != nil && baselineTop1 == candidateTop1,
            prefillTopKOverlapCount: prefillOverlap,
            prefillTopKOverlapRatio: overlapRatio(
                overlapCount: prefillOverlap,
                baselineCount: baseline.prefillTopK.count,
                candidateCount: candidate.prefillTopK.count
            ),
            decodeTopKOverlapCount: decodeOverlap?.count,
            decodeTopKOverlapRatio: decodeOverlap?.ratio
        )
    }

    private static func decodeOverlapComparison(
        baseline: LogitsDiagnosticPoint,
        candidate: LogitsDiagnosticPoint
    ) -> (count: Int, ratio: Double)? {
        guard !baseline.decodeTopK.isEmpty || !candidate.decodeTopK.isEmpty else {
            return nil
        }

        let count = overlapCount(
            baseline.decodeTopK.map(\.tokenID),
            candidate.decodeTopK.map(\.tokenID)
        )
        return (
            count,
            overlapRatio(
                overlapCount: count,
                baselineCount: baseline.decodeTopK.count,
                candidateCount: candidate.decodeTopK.count
            )
        )
    }

    private static func overlapCount(_ baseline: [Int32], _ candidate: [Int32]) -> Int {
        let baselineSet = Set(baseline)
        return Set(candidate).filter { baselineSet.contains($0) }.count
    }

    private static func overlapRatio(
        overlapCount: Int,
        baselineCount: Int,
        candidateCount: Int
    ) -> Double {
        let denominator = max(baselineCount, candidateCount)
        guard denominator > 0 else {
            return 1.0
        }

        return roundedSensitivityRatio(Double(overlapCount) / Double(denominator))
    }
}

private extension QuantizationSensitivitySummary {
    init(
        comparisons: [QuantizationSensitivityComparison],
        baseline: [LogitsDiagnosticPoint],
        candidate: [LogitsDiagnosticPoint]
    ) {
        let top1MatchCount = comparisons.filter(\.prefillTop1Matches).count
        self.init(
            baselinePointCount: baseline.count,
            candidatePointCount: candidate.count,
            comparedPointCount: comparisons.count,
            averagePrefillTopKOverlapRatio: roundedSensitivityAverage(comparisons.map(\.prefillTopKOverlapRatio)),
            prefillTop1Agreement: comparisons.isEmpty
                ? 0
                : roundedSensitivityRatio(Double(top1MatchCount) / Double(comparisons.count)),
            firstZeroPrefillOverlapPrefixTokenCount: comparisons
                .filter { $0.prefillTopKOverlapCount == 0 }
                .map(\.prefixTokenCount)
                .min()
        )
    }
}

private extension QuantizationSensitivityGate {
    init(
        summary: QuantizationSensitivitySummary,
        comparisons: [QuantizationSensitivityComparison],
        targets: QuantizationSensitivityTargets
    ) {
        var failures: [String] = []
        if summary.averagePrefillTopKOverlapRatio < targets.minimumAveragePrefillTopKOverlapRatio {
            failures.append(
                "average prefill top-k overlap \(summary.averagePrefillTopKOverlapRatio) is below \(targets.minimumAveragePrefillTopKOverlapRatio) target"
            )
        }

        for comparison in comparisons where comparison.prefixTokenCount <= targets.criticalPrefixTokenCount {
            if comparison.prefillTopKOverlapCount < targets.minimumCriticalPrefixOverlapCount {
                failures.append(
                    "prefix \(comparison.prefixTokenCount) prefill overlap \(comparison.prefillTopKOverlapCount) is below \(targets.minimumCriticalPrefixOverlapCount) critical-prefix target"
                )
            }
        }

        self.init(ok: failures.isEmpty, failures: failures, targets: targets)
    }
}

private func diagnosticKey(for point: LogitsDiagnosticPoint) -> String {
    "\(point.promptID)#\(point.prefixTokenCount)"
}

private func roundedSensitivityAverage(_ values: [Double]) -> Double {
    guard !values.isEmpty else {
        return 0
    }

    return roundedSensitivityRatio(values.reduce(0, +) / Double(values.count))
}

private func roundedSensitivityRatio(_ value: Double) -> Double {
    (value * 100).rounded() / 100
}
