package com.solarisbank.frauddetection.dto;

import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Represents a raw tokenization request payload submitted to the fraud evaluation endpoint.
 *
 * All fields are nullable — the EncodingService applies Snowflake-aligned null-handling
 * defaults for any missing or malformed values before passing the feature vector to the
 * ONNX model.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class TokenizationRequest {

    /** Internal user identifier. Used to derive a normalised hash feature. */
    private String userId;

    /** Full name of the cardholder as provided in the tokenization request. */
    private String cardholderName;

    /** Logical name or label assigned to the card token request. */
    private String cardTokenName;

    /**
     * Operating system of the requesting device.
     * Accepted values: "iOS", "Android", "Unknown".
     * Any unrecognised value is treated as "Unknown" by the EncodingService.
     */
    private String deviceOs;

    /** IPv4 address of the originating request. Used to derive private/public and octet features. */
    private String ipAddress;

    /** Geographic latitude of the requesting device. Defaults to 0.0 if null. */
    private Double latitude;

    /** Geographic longitude of the requesting device. Defaults to 0.0 if null. */
    private Double longitude;

    /**
     * ISO-8601 timestamp of the tokenization request (e.g. "2025-06-25T14:30:00Z").
     * Used to extract hour_of_day and day_of_week features.
     * Defaults to -1.0 for both derived features if null or unparseable.
     */
    @JsonFormat(shape = JsonFormat.Shape.STRING)
    private String timestamp;
}
