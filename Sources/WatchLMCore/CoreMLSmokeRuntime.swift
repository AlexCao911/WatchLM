#if canImport(CoreML)
import CoreML
import Foundation

public final class CoreMLSmokeRuntime: InferenceRuntime, @unchecked Sendable {
    private let modelURL: URL
    private let inputName: String
    private let outputName: String
    private let lock = NSLock()
    private var loadedModel: MLModel?

    public init(modelURL: URL, inputName: String, outputName: String) {
        self.modelURL = modelURL
        self.inputName = inputName
        self.outputName = outputName
    }

    public func load() async throws -> RuntimeTiming {
        let started = Date()
        let model = try loadModel()
        lock.withLock {
            loadedModel = model
        }

        return RuntimeTiming(loadMs: elapsedMilliseconds(since: started))
    }

    public func generate(
        request: InferenceRequest,
        shouldCancel: @Sendable () -> Bool
    ) async throws -> InferenceResult {
        if shouldCancel() {
            throw InferenceRuntimeError.cancelled(partialTokens: [])
        }

        let inputValue = try parseScalarPrompt(request.prompt)
        let model = try currentModel()
        let predictionStarted = Date()
        let outputValue: Double

        do {
            let input = try SingleArrayFeatureProvider(name: inputName, value: inputValue)
            #if os(watchOS)
            let output = try await model.prediction(from: input)
            #else
            let output = try model.prediction(from: input)
            #endif
            outputValue = try scalarOutput(from: output, name: outputName)
        } catch let error as InferenceRuntimeError {
            throw error
        } catch {
            throw InferenceRuntimeError.predictionFailed(message: error.localizedDescription)
        }

        if shouldCancel() {
            throw InferenceRuntimeError.cancelled(partialTokens: [])
        }

        let predictionMs = elapsedMilliseconds(since: predictionStarted)
        return InferenceResult(
            tokens: [formatToken(outputValue)],
            timing: RuntimeTiming(
                firstTokenMs: predictionMs,
                decodeStepMs: [predictionMs]
            )
        )
    }

    private func currentModel() throws -> MLModel {
        if let model = lock.withLock({ loadedModel }) {
            return model
        }

        let model = try loadModel()
        lock.withLock {
            loadedModel = model
        }
        return model
    }

    private func loadModel() throws -> MLModel {
        do {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .all
            return try MLModel(contentsOf: modelURL, configuration: configuration)
        } catch {
            throw InferenceRuntimeError.unavailableRuntime(reason: "Core ML load failed: \(error.localizedDescription)")
        }
    }
}

private final class SingleArrayFeatureProvider: MLFeatureProvider {
    let featureNames: Set<String>
    private let name: String
    private let featureValue: MLFeatureValue

    init(name: String, value: Double) throws {
        self.name = name
        self.featureNames = [name]

        let array = try MLMultiArray(shape: [1], dataType: .double)
        array[0] = NSNumber(value: value)
        self.featureValue = MLFeatureValue(multiArray: array)
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        featureName == name ? featureValue : nil
    }
}

private func parseScalarPrompt(_ prompt: String) throws -> Double {
    let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
    guard let value = Double(trimmed), value.isFinite else {
        throw InferenceRuntimeError.invalidInput(message: "Core ML smoke runtime expects a numeric scalar prompt.")
    }

    return value
}

private func scalarOutput(from output: MLFeatureProvider, name: String) throws -> Double {
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

private func formatToken(_ value: Double) -> String {
    String(format: "%.6g", value)
}

private func elapsedMilliseconds(since started: Date) -> Double {
    let elapsed = Date().timeIntervalSince(started) * 1000
    return (elapsed * 1000).rounded() / 1000
}
#endif
