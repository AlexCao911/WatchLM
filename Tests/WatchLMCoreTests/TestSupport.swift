import Foundation
@testable import WatchLMCore

func loadSampleManifest() throws -> ModelManifest {
    let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let url = root.appendingPathComponent("fixtures/sample-model-manifest.json")
    let data = try Data(contentsOf: url)
    return try JSONDecoder().decode(ModelManifest.self, from: data)
}
