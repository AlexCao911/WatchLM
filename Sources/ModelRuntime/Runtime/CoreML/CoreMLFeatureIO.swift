#if canImport(CoreML)
import CoreML

final class CoreMLDictionaryFeatureProvider: MLFeatureProvider {
    private let features: [String: MLFeatureValue]

    var featureNames: Set<String> {
        Set(features.keys)
    }

    init(features: [String: MLFeatureValue]) throws {
        guard !features.isEmpty else {
            throw InferenceRuntimeError.invalidInput(message: "Core ML input features cannot be empty.")
        }

        self.features = features
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        features[featureName]
    }
}

func tokenIDOutput(from output: MLFeatureProvider, name: String) throws -> Int32 {
    let value = try scalarCoreMLOutput(from: output, name: name)
    guard value.isFinite, value >= 0, value <= Double(Int32.max) else {
        throw InferenceRuntimeError.predictionFailed(message: "Output token \(name) is outside Int32 range.")
    }

    return Int32(value.rounded())
}

func multiArrayOutput(from output: MLFeatureProvider, name: String) throws -> MLMultiArray {
    guard let value = output.featureValue(for: name) else {
        throw InferenceRuntimeError.predictionFailed(message: "Missing output feature \(name).")
    }

    guard let multiArray = value.multiArrayValue else {
        throw InferenceRuntimeError.predictionFailed(message: "Output feature \(name) is not an MLMultiArray.")
    }

    return multiArray
}

private func scalarCoreMLOutput(from output: MLFeatureProvider, name: String) throws -> Double {
    guard let value = output.featureValue(for: name) else {
        throw InferenceRuntimeError.predictionFailed(message: "Missing output feature \(name).")
    }

    if let multiArray = value.multiArrayValue {
        guard multiArray.count > 0 else {
            throw InferenceRuntimeError.predictionFailed(message: "Output feature \(name) is empty.")
        }
        return multiArray[0].doubleValue
    }

    if value.type == .double {
        return value.doubleValue
    }

    if value.type == .int64 {
        return Double(value.int64Value)
    }

    throw InferenceRuntimeError.predictionFailed(message: "Output feature \(name) is not numeric.")
}
#endif
