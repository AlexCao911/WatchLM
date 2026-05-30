import Foundation
import Testing
@testable import WatchLMCore

@Test func coreMLRuntimeCapabilitiesGateStatefulPredictionByPlatformVersion() {
    #expect(CoreMLRuntimeCapabilities.inferredStatefulPredictionSupport(
        platform: .watchOS,
        operatingSystemVersion: OperatingSystemVersion(majorVersion: 10, minorVersion: 6, patchVersion: 0)
    ) == false)
    #expect(CoreMLRuntimeCapabilities.inferredStatefulPredictionSupport(
        platform: .watchOS,
        operatingSystemVersion: OperatingSystemVersion(majorVersion: 11, minorVersion: 0, patchVersion: 0)
    ))
    #expect(CoreMLRuntimeCapabilities.inferredStatefulPredictionSupport(
        platform: .macOS,
        operatingSystemVersion: OperatingSystemVersion(majorVersion: 15, minorVersion: 0, patchVersion: 0)
    ))
}

@Test func coreMLKVCacheRoutePlannerSelectsStatefulOnlyWhenSupported() {
    let watchOS10 = CoreMLRuntimeCapabilities(
        platform: .watchOS,
        operatingSystemVersion: OperatingSystemVersion(majorVersion: 10, minorVersion: 6, patchVersion: 0)
    )
    let watchOS11 = CoreMLRuntimeCapabilities(
        platform: .watchOS,
        operatingSystemVersion: OperatingSystemVersion(majorVersion: 11, minorVersion: 0, patchVersion: 0)
    )

    let fallback = CoreMLKVCacheRoutePlanner.selectRoute(
        kvCacheMode: "stateful-preferred",
        capabilities: watchOS10
    )
    let stateful = CoreMLKVCacheRoutePlanner.selectRoute(
        kvCacheMode: "stateful-preferred",
        capabilities: watchOS11
    )
    let slotRing = CoreMLKVCacheRoutePlanner.selectRoute(
        kvCacheMode: "slot-ring",
        capabilities: watchOS11
    )
    let explicitGraph = CoreMLKVCacheRoutePlanner.selectRoute(
        kvCacheMode: "stateful-preferred",
        graphInterface: "logits-layered-kv",
        capabilities: watchOS11
    )
    let unsupportedStatefulGraph = CoreMLKVCacheRoutePlanner.selectRoute(
        kvCacheMode: "stateful-preferred",
        graphInterface: "stateful-kv",
        capabilities: watchOS10
    )

    #expect(fallback.selectedRoute == .explicitSlotRing)
    #expect(fallback.reason.contains("falling back"))
    #expect(stateful.selectedRoute == .statefulKV)
    #expect(slotRing.selectedRoute == .explicitSlotRing)
    #expect(explicitGraph.selectedRoute == .explicitSlotRing)
    #expect(explicitGraph.reason.contains("exposes explicit KV tensors"))
    #expect(unsupportedStatefulGraph.selectedRoute == .unsupportedStatefulKV)
}
