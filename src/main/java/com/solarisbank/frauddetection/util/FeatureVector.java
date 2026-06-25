package com.solarisbank.frauddetection.util;

/**
 * Canonical feature vector definition for the fraud detection ONNX model.
 *
 * This class defines the strict index order of every feature in the float[]
 * array passed to ModelInferenceService. The order here is the contract between
 * EncodingService and the trained model — it must match the column order used
 * during Snowflake data export and model training exactly.
 *
 * Feature vector length: {@value FEATURE_COUNT}
 *
 * Source fields are derived from {@link com.solarisbank.frauddetection.dto.TokenizationRequest}
 * and its nested objects ({@link com.solarisbank.frauddetection.dto.TokenInfo},
 * {@link com.solarisbank.frauddetection.dto.DeviceInfo}).
 *
 * Encoding rules:
 * - One-hot binary fields:  1.0f (present) or 0.0f (absent)
 * - Temporal fields:        raw int cast to float; -1.0f if unavailable
 * - Numeric fields:         raw value cast to float; 0.0f if null
 * - Normalised fields:      see individual field javadoc for formula
 *
 * Default fallbacks align with Snowflake cleaning guidelines to prevent
 * training-serving skew.
 */
public final class FeatureVector {

    private FeatureVector() {}

    /** Total number of features in the vector. Must match the ONNX model input shape [1, FEATURE_COUNT]. */
    public static final int FEATURE_COUNT = 10;

    // -------------------------------------------------------------------------
    // Device OS — one-hot encoding derived from deviceInfo.osType
    // -------------------------------------------------------------------------

    /** Index 0: 1.0f if deviceInfo.osType == "iOS" (case-insensitive), else 0.0f */
    public static final int DEVICE_OS_IOS = 0;

    /** Index 1: 1.0f if deviceInfo.osType == "Android" (case-insensitive), else 0.0f */
    public static final int DEVICE_OS_ANDROID = 1;

    // -------------------------------------------------------------------------
    // Temporal features — derived from server-side request ingestion time (UTC)
    // -------------------------------------------------------------------------

    /** Index 2: Hour of day in UTC at time of request ingestion (0–23). Default: -1.0f on error. */
    public static final int HOUR_OF_DAY = 2;

    /** Index 3: Day of week at time of request ingestion (1=Monday … 7=Sunday, ISO-8601). Default: -1.0f on error. */
    public static final int DAY_OF_WEEK = 3;

    // -------------------------------------------------------------------------
    // IP address derived features — from deviceInfo.deviceIpAddressV4
    // -------------------------------------------------------------------------

    /** Index 4: First octet of deviceIpAddressV4 as float (e.g. "192.168.1.1" → 192.0f). Default: 0.0f if null or malformed. */
    public static final int IP_FIRST_OCTET = 4;

    /**
     * Index 5: 1.0f if deviceIpAddressV4 falls within an RFC-1918 private range
     * (10.x.x.x, 172.16–31.x.x, 192.168.x.x), else 0.0f.
     * Default: 0.0f if null or malformed.
     */
    public static final int IP_IS_PRIVATE = 5;

    // -------------------------------------------------------------------------
    // Token count features — from tokenInfo
    // -------------------------------------------------------------------------

    /**
     * Index 6: Normalised count of active tokens for this PAN.
     * Computed as: min(tokenInfo.numberOfActiveTokensForPAN, 20) / 20.0f
     * Default: 0.0f if tokenInfo or field is null.
     */
    public static final int NUM_ACTIVE_TOKENS_NORM = 6;

    /**
     * Index 7: Normalised count of suspended tokens for this PAN.
     * Computed as: min(tokenInfo.numberOfSuspendedTokensForPAN, 20) / 20.0f
     * Default: 0.0f if tokenInfo or field is null.
     */
    public static final int NUM_SUSPENDED_TOKENS_NORM = 7;

    // -------------------------------------------------------------------------
    // Identity-derived features
    // -------------------------------------------------------------------------

    /**
     * Index 8: Normalised hash of clientWalletAccountId.
     * Computed as: abs(clientWalletAccountId.hashCode()) % 1000 / 1000.0f
     * Default: 0.0f if null or blank.
     */
    public static final int CLIENT_WALLET_ACCOUNT_HASH_NORM = 8;

    /**
     * Index 9: Normalised length of tokenReferenceId.
     * Computed as: min(tokenReferenceId.length(), 100) / 100.0f
     * Default: 0.0f if null or blank.
     */
    public static final int TOKEN_REFERENCE_ID_LEN_NORM = 9;
}
