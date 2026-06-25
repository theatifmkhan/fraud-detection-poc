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
 * Encoding rules:
 * - One-hot binary fields:  1.0f (present) or 0.0f (absent)
 * - Temporal fields:        raw int cast to float; -1.0f if null/unparseable
 * - Numeric fields:         raw double cast to float; 0.0f if null
 * - Derived fields:         see individual field javadoc
 *
 * Default fallbacks align with Snowflake cleaning guidelines to prevent
 * training-serving skew.
 */
public final class FeatureVector {

    private FeatureVector() {}

    /** Total number of features in the vector. Must match the ONNX model input shape [1, FEATURE_COUNT]. */
    public static final int FEATURE_COUNT = 10;

    // -------------------------------------------------------------------------
    // Device OS — one-hot encoding (iOS, Android; both 0 = Unknown)
    // -------------------------------------------------------------------------

    /** Index 0: 1.0f if deviceOs == "iOS", else 0.0f */
    public static final int DEVICE_OS_IOS = 0;

    /** Index 1: 1.0f if deviceOs == "Android", else 0.0f */
    public static final int DEVICE_OS_ANDROID = 1;

    // -------------------------------------------------------------------------
    // Temporal features — extracted from ISO-8601 timestamp (UTC)
    // -------------------------------------------------------------------------

    /** Index 2: Hour of day in UTC (0–23). Default: -1.0f if timestamp is null or unparseable. */
    public static final int HOUR_OF_DAY = 2;

    /** Index 3: Day of week (1=Monday … 7=Sunday, ISO-8601). Default: -1.0f if timestamp is null or unparseable. */
    public static final int DAY_OF_WEEK = 3;

    // -------------------------------------------------------------------------
    // Geolocation features
    // -------------------------------------------------------------------------

    /** Index 4: Raw latitude as float. Default: 0.0f if null. */
    public static final int LATITUDE = 4;

    /** Index 5: Raw longitude as float. Default: 0.0f if null. */
    public static final int LONGITUDE = 5;

    // -------------------------------------------------------------------------
    // IP address derived features
    // -------------------------------------------------------------------------

    /** Index 6: First octet of the IPv4 address as float (e.g. 192.168.1.1 → 192.0f). Default: 0.0f if null or malformed. */
    public static final int IP_FIRST_OCTET = 6;

    /**
     * Index 7: 1.0f if the IP address falls within an RFC-1918 private range
     * (10.x.x.x, 172.16–31.x.x, 192.168.x.x), else 0.0f.
     * Default: 0.0f if null or malformed.
     */
    public static final int IP_IS_PRIVATE = 7;

    // -------------------------------------------------------------------------
    // Identity-derived features
    // -------------------------------------------------------------------------

    /**
     * Index 8: Normalised hash of userId.
     * Computed as: abs(userId.hashCode()) % 1000 / 1000.0f
     * Default: 0.0f if userId is null or blank.
     */
    public static final int USER_ID_HASH_NORM = 8;

    /**
     * Index 9: Normalised length of cardTokenName.
     * Computed as: min(cardTokenName.length(), 100) / 100.0f
     * Default: 0.0f if cardTokenName is null or blank.
     */
    public static final int CARD_TOKEN_NAME_LEN_NORM = 9;
}
