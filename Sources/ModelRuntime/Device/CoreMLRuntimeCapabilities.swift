import Foundation

public enum CoreMLRuntimePlatform: String, Codable, Equatable, Sendable {
    case macOS
    case iOS
    case watchOS
    case tvOS
    case visionOS
    case unknown

    public static var current: CoreMLRuntimePlatform {
        #if os(macOS)
        return .macOS
        #elseif os(iOS)
        return .iOS
        #elseif os(watchOS)
        return .watchOS
        #elseif os(tvOS)
        return .tvOS
        #elseif os(visionOS)
        return .visionOS
        #else
        return .unknown
        #endif
    }
}

public struct CoreMLRuntimeOSVersion: Codable, Comparable, Sendable {
    public var majorVersion: Int
    public var minorVersion: Int
    public var patchVersion: Int

    public init(majorVersion: Int, minorVersion: Int, patchVersion: Int) {
        self.majorVersion = majorVersion
        self.minorVersion = minorVersion
        self.patchVersion = patchVersion
    }

    public init(_ version: OperatingSystemVersion) {
        self.init(
            majorVersion: version.majorVersion,
            minorVersion: version.minorVersion,
            patchVersion: version.patchVersion
        )
    }

    public static func < (lhs: CoreMLRuntimeOSVersion, rhs: CoreMLRuntimeOSVersion) -> Bool {
        if lhs.majorVersion != rhs.majorVersion {
            return lhs.majorVersion < rhs.majorVersion
        }
        if lhs.minorVersion != rhs.minorVersion {
            return lhs.minorVersion < rhs.minorVersion
        }
        return lhs.patchVersion < rhs.patchVersion
    }

    public var displayString: String {
        "\(majorVersion).\(minorVersion).\(patchVersion)"
    }
}

public struct CoreMLRuntimeCapabilities: Codable, Equatable, Sendable {
    public var platform: CoreMLRuntimePlatform
    public var operatingSystemVersion: CoreMLRuntimeOSVersion
    public var supportsStatefulPrediction: Bool

    public init(
        platform: CoreMLRuntimePlatform,
        operatingSystemVersion: OperatingSystemVersion,
        supportsStatefulPrediction: Bool? = nil
    ) {
        self.init(
            platform: platform,
            operatingSystemVersion: CoreMLRuntimeOSVersion(operatingSystemVersion),
            supportsStatefulPrediction: supportsStatefulPrediction
        )
    }

    public init(
        platform: CoreMLRuntimePlatform,
        operatingSystemVersion: CoreMLRuntimeOSVersion,
        supportsStatefulPrediction: Bool? = nil
    ) {
        self.platform = platform
        self.operatingSystemVersion = operatingSystemVersion
        self.supportsStatefulPrediction = supportsStatefulPrediction
            ?? Self.inferredStatefulPredictionSupport(platform: platform, operatingSystemVersion: operatingSystemVersion)
    }

    public static var current: CoreMLRuntimeCapabilities {
        CoreMLRuntimeCapabilities(
            platform: CoreMLRuntimePlatform.current,
            operatingSystemVersion: ProcessInfo.processInfo.operatingSystemVersion
        )
    }

    public static func inferredStatefulPredictionSupport(
        platform: CoreMLRuntimePlatform,
        operatingSystemVersion: OperatingSystemVersion
    ) -> Bool {
        inferredStatefulPredictionSupport(
            platform: platform,
            operatingSystemVersion: CoreMLRuntimeOSVersion(operatingSystemVersion)
        )
    }

    public static func inferredStatefulPredictionSupport(
        platform: CoreMLRuntimePlatform,
        operatingSystemVersion: CoreMLRuntimeOSVersion
    ) -> Bool {
        guard let minimumVersion = minimumStatefulPredictionVersion(for: platform) else {
            return false
        }
        return operatingSystemVersion >= minimumVersion
    }

    public static func minimumStatefulPredictionVersion(
        for platform: CoreMLRuntimePlatform
    ) -> CoreMLRuntimeOSVersion? {
        switch platform {
        case .macOS:
            CoreMLRuntimeOSVersion(majorVersion: 15, minorVersion: 0, patchVersion: 0)
        case .iOS, .tvOS:
            CoreMLRuntimeOSVersion(majorVersion: 18, minorVersion: 0, patchVersion: 0)
        case .watchOS:
            CoreMLRuntimeOSVersion(majorVersion: 11, minorVersion: 0, patchVersion: 0)
        case .visionOS:
            CoreMLRuntimeOSVersion(majorVersion: 2, minorVersion: 0, patchVersion: 0)
        case .unknown:
            nil
        }
    }
}

public enum CoreMLKVCacheRuntimeRoute: String, Codable, Equatable, Sendable {
    case statefulKV
    case explicitSlotRing
    case explicitContiguousSliding

    public var explicitUpdateStrategy: KVCacheUpdateStrategy? {
        switch self {
        case .statefulKV:
            nil
        case .explicitSlotRing:
            .slotRing
        case .explicitContiguousSliding:
            .contiguousSliding
        }
    }
}

public struct CoreMLKVCacheRouteDecision: Codable, Equatable, Sendable {
    public var requestedMode: String
    public var selectedRoute: CoreMLKVCacheRuntimeRoute
    public var reason: String

    public init(
        requestedMode: String,
        selectedRoute: CoreMLKVCacheRuntimeRoute,
        reason: String
    ) {
        self.requestedMode = requestedMode
        self.selectedRoute = selectedRoute
        self.reason = reason
    }
}

public enum CoreMLKVCacheRoutePlanner {
    public static func selectRoute(
        kvCacheMode: String,
        graphInterface: String? = nil,
        capabilities: CoreMLRuntimeCapabilities
    ) -> CoreMLKVCacheRouteDecision {
        switch kvCacheMode {
        case "stateful-preferred":
            if let graphInterface, graphInterface != "stateful-kv" {
                return CoreMLKVCacheRouteDecision(
                    requestedMode: kvCacheMode,
                    selectedRoute: .explicitSlotRing,
                    reason: "Artifact graph interface \(graphInterface) exposes explicit KV tensors; using explicit slot-ring KV."
                )
            }

            if capabilities.supportsStatefulPrediction {
                return CoreMLKVCacheRouteDecision(
                    requestedMode: kvCacheMode,
                    selectedRoute: .statefulKV,
                    reason: "Core ML stateful prediction is available on \(capabilities.platform.rawValue) \(capabilities.operatingSystemVersion.displayString)."
                )
            }

            return CoreMLKVCacheRouteDecision(
                requestedMode: kvCacheMode,
                selectedRoute: .explicitSlotRing,
                reason: "Core ML stateful prediction requires a newer OS on \(capabilities.platform.rawValue); falling back to explicit slot-ring KV."
            )
        case "slot-ring":
            return CoreMLKVCacheRouteDecision(
                requestedMode: kvCacheMode,
                selectedRoute: .explicitSlotRing,
                reason: "Manifest requested explicit slot-ring KV."
            )
        case "contiguous-sliding":
            return CoreMLKVCacheRouteDecision(
                requestedMode: kvCacheMode,
                selectedRoute: .explicitContiguousSliding,
                reason: "Manifest requested explicit contiguous-sliding KV."
            )
        default:
            return CoreMLKVCacheRouteDecision(
                requestedMode: kvCacheMode,
                selectedRoute: .explicitSlotRing,
                reason: "Unsupported KV cache mode \(kvCacheMode); falling back to explicit slot-ring KV."
            )
        }
    }
}
