public enum ContextVariantSelectionError: Error, Equatable, Sendable {
    case unsupportedDeviceProfile(String)
    case noContextVariants
}

public enum ContextVariantSelector {
    public static func select(
        from manifest: ModelManifest,
        for deviceProfile: DeviceProfile,
        requestedTokens: Int? = nil
    ) throws -> Int {
        guard let profile = manifest.deviceProfiles[deviceProfile.rawValue] else {
            throw ContextVariantSelectionError.unsupportedDeviceProfile(deviceProfile.rawValue)
        }

        let variants = manifest.contextVariants.sorted()
        guard let smallest = variants.first else {
            throw ContextVariantSelectionError.noContextVariants
        }

        let requested = requestedTokens ?? profile.defaultContextVariant
        return variants.last(where: { $0 <= requested }) ?? smallest
    }
}
