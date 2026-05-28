// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "WatchLM",
    platforms: [
        .macOS(.v13),
        .watchOS(.v10)
    ],
    products: [
        .library(name: "WatchLMCore", targets: ["WatchLMCore"])
    ],
    targets: [
        .target(name: "WatchLMCore"),
        .testTarget(
            name: "WatchLMCoreTests",
            dependencies: ["WatchLMCore"]
        )
    ]
)
