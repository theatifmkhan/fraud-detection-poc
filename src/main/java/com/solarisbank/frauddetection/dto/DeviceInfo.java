package com.solarisbank.frauddetection.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Represents the deviceInfo object from the Visa approveProvisioning API.
 * All fields are optional — EncodingService applies null-safe defaults.
 *
 * Source: visa-tokenization/source/app/api/visa/approve_provisioning.rb
 * Canonical field list from: spec/unit/.../token_provisioning_request_spec.rb
 *
 * Key fields used by EncodingService for feature extraction:
 * - osType          → device_os_ios / device_os_android (one-hot)
 * - deviceIpAddressV4 → ip_first_octet, ip_is_private
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class DeviceInfo {

    private String deviceId;
    private String deviceLanguageCode;

    /**
     * Type of the requesting device (e.g. "MOBILE_PHONE", "TABLET").
     * Used in Visa green-path logic (device_type == "MOBILE_PHONE").
     */
    private String deviceType;

    private String deviceName;

    /**
     * Operating system type of the device.
     * Used by EncodingService for device_os_ios / device_os_android features.
     * Expected values: "iOS", "Android" — anything else encodes as [0, 0].
     */
    private String osType;

    private String deviceIdType;
    private String deviceManufacturer;
    private String deviceBrand;
    private String deviceLocation;
    private String deviceIndex;

    /**
     * IPv4 address of the requesting device.
     * Used by EncodingService to derive ip_first_octet and ip_is_private features.
     */
    private String deviceIpAddressV4;

    private String tokenProtectionMethod;
    private String originalDeviceId;
    private String originalDeviceIdType;

    /**
     * Mobile phone number of the device owner.
     * Used in the Visa workflow for Apple Pay mobile number verification.
     */
    private String deviceNumber;
}
