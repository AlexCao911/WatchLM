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
        .target(name: "WatchLMCore", path: "Sources/ModelRuntime"),
        .testTarget(
            name: "WatchLMCoreTests",
            dependencies: ["WatchLMCore"],
            exclude: [
                "Resources/SmokeDecode.mlmodel",
                "Resources/SmokeIdentity.mlmodel",
                "Resources/SmokeLayeredDecode.mlpackage",
                "Resources/SmokeLayeredPrefill.mlpackage",
                "Resources/SmokePrefill.mlmodel"
            ],
            resources: [
                .copy("Resources/SmokeDecode_macOS.mlmodelc"),
                .copy("Resources/SmokeDecode_watchOS.mlmodelc"),
                .copy("Resources/SmokeIdentity_macOS.mlmodelc"),
                .copy("Resources/SmokeIdentity_watchOS.mlmodelc"),
                .copy("Resources/SmokeLayeredDecode_macOS.mlmodelc"),
                .copy("Resources/SmokeLayeredDecode_watchOS.mlmodelc"),
                .copy("Resources/SmokeLayeredPrefill_macOS.mlmodelc"),
                .copy("Resources/SmokeLayeredPrefill_watchOS.mlmodelc"),
                .copy("Resources/SmokePrefill_macOS.mlmodelc"),
                .copy("Resources/SmokePrefill_watchOS.mlmodelc")
            ]
        )
    ]
)
