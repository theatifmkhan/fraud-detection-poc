package com.solarisbank.frauddetection.util;

/**
 * Canonical feature vector definition for the fraud detection ONNX model.
 *
 * Defines the strict index order of all 26 features in the float[] array
 * passed to ModelInferenceService. This contract must match the column order
 * used during Snowflake data export and SageMaker model training exactly.
 *
 * Feature vector length: {@value FEATURE_COUNT}
 *
 * Source document: Shack.xlsx — "DataPoints Definition" sheet (26 rows).
 *
 * ──────────────────────────────────────────────────────────────────────
 * Encoding conventions by data type:
 *
 *   High-Cardinality / Free Text
 *     Snowflake-compatible MD5 hash:
 *       ABS(MD5_NUMBER_LOWER64(value)) % buckets
 *     Implemented via EncodingService.snowflakeHash(value, buckets).
 *     Normalised by dividing by the bucket count.
 *     Default 0.0f when null or blank.
 *
 *   IP Address (High-Cardinality)
 *     As above, but input is the first 3 octets of the IPv4 address
 *     (e.g. "192.168.1" from "192.168.1.55"), with 10000 buckets.
 *     Normalised by /10000.
 *
 *   Low-Cardinality (categorical)
 *     Integer label from encoding-mappings.json.
 *     Normalised by dividing by (number_of_labels - 1).
 *     Unknown / null values default to 0.0f.
 *
 *   Ordinal
 *     Raw integer cast to float. Normalised by the declared max value.
 *     Default 0.0f when null.
 * ──────────────────────────────────────────────────────────────────────
 */
public final class FeatureVector {

    private FeatureVector() {}

    /** Total number of features. Must match the ONNX model input shape [1, FEATURE_COUNT]. */
    public static final int FEATURE_COUNT = 26;

    // ─────────────────────────────────────────────────────────────────────────
    // Index 0 — account_holder_name
    // ─────────────────────────────────────────────────────────────────────────
    /**
     * Free Text | Snowflake MD5 % 1000 | normalised /1000
     * Source: {@code TokenizationRequest.accountHolderName}
     */
    public static final int ACCOUNT_HOLDER_NAME = 0;

    // ─────────────────────────────────────────────────────────────────────────
    // Index 1 — token_requestor_id
    // ─────────────────────────────────────────────────────────────────────────
    /**
     * Low-Cardinality | Label: APPLE=0, GOOGLE=1, OTHER=2 | normalised /2
     * Source: {@code TokenizationRequest.tokenRequestorId}
     */
    public static final int TOKEN_REQUESTOR_ID = 1;

    // ─────────────────────────────────────────────────────────────────────────
    // Index 2 — consumer_entry_mode
    // ─────────────────────────────────────────────────────────────────────────
    /**
     * Low-Cardinality | Label: KEY_ENTERED=0, CAMERA_CAPTURED=1, UNKNOWN=2 | normalised /2
     * Source: {@code TokenizationRequest.consumerEntryMode}
     */
    public static final int CONSUMER_ENTRY_MODE = 2;

    // ─────────────────────────────────────────────────────────────────────────
    // Index 3 — device_ip_address_v4
    // ─────────────────────────────────────────────────────────────────────────
    /**
     * High-Cardinality IP | MD5(first 3 octets) % 10000 | normalised /10000
     * Source: {@code TokenizationRequest.deviceInfo.deviceIpAddressV4}
     */
    public static final int DEVICE_IP_ADDRESS_V4 = 3;

    // ─────────────────────────────────────────────────────────────────────────
    // Index 4 — wallet_provider_device_score
    // ─────────────────────────────────────────────────────────────────────────
    /**
     * Ordinal 1–5 | normalised /5 | default 0.0f
     * Source: {@code TokenizationRequest.walletProviderDeviceScore}
     */
    public static final int WALLET_PROVIDER_DEVICE_SCORE = 4;

    // ─────────────────────────────────────────────────────────────────────────
    // Index 5 — device_type
    // ─────────────────────────────────────────────────────────────────────────
    /**
     * Low-Cardinality | Label: UNKNOWN=0, MOBILE_PHONE=1, TABLET=2, WATCH=3,
     * MOBILEPHONE_OR_TABLET=4, PC=5, HOUSEHOLD_DEVICE=6, WEARABLE_DEVICE=7,
     * AUTOMOBILE_DEVICE=8 | normalised /8
     * Source: {@code TokenizationRequest.deviceInfo.deviceType}
     */
    public static final int DEVICE_TYPE = 5;

    // ─────────────────────────────────────────────────────────────────────────
    // Index 6 — wallet_provider_account_score
    // ─────────────────────────────────────────────────────────────────────────
    /**
     * Ordinal 1–5 | normalised /5 | default 0.0f
     * Source: {@code TokenizationRequest.walletProviderAccountScore}
     */
    public static final int WALLET_PROVIDER_ACCOUNT_SCORE = 6;

    // ─────────────────────────────────────────────────────────────────────────
    // Index 7 — device_name
    // ─────────────────────────────────────────────────────────────────────────
    /**
     * Free Text | Snowflake MD5 % 1000 | normalised /1000
     * Source: {@code TokenizationRequest.deviceInfo.deviceName}
     */
    public static final int DEVICE_NAME = 7;

    // ─────────────────────────────────────────────────────────────────────────
    // Index 8 — device_id
    // ─────────────────────────────────────────────────────────────────────────
    /**
     * High-Cardinality | Snowflake MD5 % 1000 | normalised /1000
     * Source: {@code TokenizationRequest.deviceInfo.deviceId}
     */
    public static final int DEVICE_ID = 8;

    // ─────────────────────────────────────────────────────────────────────────
    // Index 9 — wallet_provider_risk_assessment
    // ─────────────────────────────────────────────────────────────────────────
    /**
     * Ordinal 0–2 | normalised /2 | default 0.0f
     * 0=approve, 1=decline, 2=require step-up
     * Source: {@code TokenizationRequest.walletProviderRiskAssessment}
     */
    public static final int WALLET_PROVIDER_RISK_ASSESSMENT = 9;

    // ─────────────────────────────────────────────────────────────────────────
    // Index 10 — device_number
    // ─────────────────────────────────────────────────────────────────────────
    /**
     * High-Cardinality | Snowflake MD5 % 1000 | normalised /1000
     * Source: {@code TokenizationRequest.deviceInfo.deviceNumber}
     */
    public static final int DEVICE_NUMBER = 10;

    // ─────────────────────────────────────────────────────────────────────────
    // Index 11 — wallet_provider_reason_codes
    // ─────────────────────────────────────────────────────────────────────────
    /**
     * Low-Cardinality multi-value | Snowflake MD5 of full string % 100 | normalised /100
     * Value may be a single code ("01") or comma-separated ("01,A5,0G").
     * Source: {@code TokenizationRequest.walletProviderReasonCodes}
     */
    public static final int WALLET_PROVIDER_REASON_CODES = 11;

    // ─────────────────────────────────────────────────────────────────────────
    // Index 12 — token_type
    // ─────────────────────────────────────────────────────────────────────────
    /**
     * Low-Cardinality | Label: SECURE_ELEMENT=0, HCE=1, CARD_ON_FILE=2,
     * ECOMMERCE=3, QRC=4 | normalised /4
     * Source: {@code TokenizationRequest.tokenInfo.tokenType}
     */
    public static final int TOKEN_TYPE = 12;

    // ─────────────────────────────────────────────────────────────────────────
    // Index 13 — risk_assessment_score
    // ─────────────────────────────────────────────────────────────────────────
    /**
     * Ordinal 1–5 | normalised /5 | default 0.0f
     * Source: {@code TokenizationRequest.riskAssessmentScore}
     */
    public static final int RISK_ASSESSMENT_SCORE = 13;

    // ─────────────────────────────────────────────────────────────────────────
    // Index 14 — device_language_code
    // ─────────────────────────────────────────────────────────────────────────
    /**
     * High-Cardinality | Snowflake MD5 % 1000 | normalised /1000
     * Source: {@code TokenizationRequest.deviceInfo.deviceLanguageCode}
     */
    public static final int DEVICE_LANGUAGE_CODE = 14;

    // ─────────────────────────────────────────────────────────────────────────
    // Index 15 — pan_reference_id
    // ─────────────────────────────────────────────────────────────────────────
    /**
     * High-Cardinality | Snowflake MD5 % 1000 | normalised /1000
     * Source: {@code TokenizationRequest.panReferenceId}
     */
    public static final int PAN_REFERENCE_ID = 15;

    // ─────────────────────────────────────────────────────────────────────────
    // Index 16 — cvv2_results_code
    // ─────────────────────────────────────────────────────────────────────────
    /**
     * Low-Cardinality | Label: M=0, N=1, P=2, S=3, U=4, NULL=5 | normalised /5
     * Source: {@code TokenizationRequest.cvv2ResultsCode}
     */
    public static final int CVV2_RESULTS_CODE = 16;

    // ─────────────────────────────────────────────────────────────────────────
    // Index 17 — pan_source
    // ─────────────────────────────────────────────────────────────────────────
    /**
     * Low-Cardinality | Label: KEY_ENTERED=0, ON_FILE=1, MOBILE_BANKING_APP=2,
     * TOKEN=3, CHIP_DIP=4, CONTACTLESS_TAP=5 | normalised /5
     * Source: {@code TokenizationRequest.panSource}
     */
    public static final int PAN_SOURCE = 17;

    // ─────────────────────────────────────────────────────────────────────────
    // Index 18 — visa_token_score
    // ─────────────────────────────────────────────────────────────────────────
    /**
     * Ordinal 1–5 | normalised /5 | default 0.0f
     * Source: {@code TokenizationRequest.visaTokenScore}
     */
    public static final int VISA_TOKEN_SCORE = 18;

    // ─────────────────────────────────────────────────────────────────────────
    // Index 19 — NAME_ON_ACCOUNT
    // ─────────────────────────────────────────────────────────────────────────
    /**
     * Free Text | Snowflake MD5 % 1000 | normalised /1000
     * Source: {@code TokenizationRequest.nameOnAccount}
     * Distinct from ACCOUNT_HOLDER_NAME (index 0) per training data schema.
     */
    public static final int NAME_ON_ACCOUNT = 19;

    // ─────────────────────────────────────────────────────────────────────────
    // Index 20 — CARDHOLDER_COUNTRY
    // ─────────────────────────────────────────────────────────────────────────
    /**
     * High-Cardinality | Snowflake MD5 % 1000 | normalised /1000
     * Source: {@code TokenizationRequest.cardholderCountry}
     */
    public static final int CARDHOLDER_COUNTRY = 20;

    // ─────────────────────────────────────────────────────────────────────────
    // Index 21 — TOKEN_PROVISION_IP_COUNTRY
    // ─────────────────────────────────────────────────────────────────────────
    /**
     * High-Cardinality | Snowflake MD5 % 10000 | normalised /10000
     * Resolved from deviceIpAddressV4 via MaxMind in production.
     * Source: {@code TokenizationRequest.tokenProvisionIpCountry}
     */
    public static final int TOKEN_PROVISION_IP_COUNTRY = 21;

    // ─────────────────────────────────────────────────────────────────────────
    // Index 22 — LAST_LOGGED_IN_DEVICE_TYPE
    // ─────────────────────────────────────────────────────────────────────────
    /**
     * Low-Cardinality | Label: iOS=0, Android=1, Web=2 | normalised /2
     * Source: {@code TokenizationRequest.lastLoggedInDeviceType}
     */
    public static final int LAST_LOGGED_IN_DEVICE_TYPE = 22;

    // ─────────────────────────────────────────────────────────────────────────
    // Index 23 — LAST_LOGGED_IN_DEVICE_NAME
    // ─────────────────────────────────────────────────────────────────────────
    /**
     * High-Cardinality | Snowflake MD5 % 1000 | normalised /1000
     * Source: {@code TokenizationRequest.lastLoggedInDeviceName}
     */
    public static final int LAST_LOGGED_IN_DEVICE_NAME = 23;

    // ─────────────────────────────────────────────────────────────────────────
    // Index 24 — LAST_LOGGED_IN_COUNTRY
    // ─────────────────────────────────────────────────────────────────────────
    /**
     * High-Cardinality | Snowflake MD5 % 1000 | normalised /1000
     * Source: DEVICE_DETAILS_DEVICE_IP_COUNTRY column.
     * Source: {@code TokenizationRequest.lastLoggedInCountry}
     */
    public static final int LAST_LOGGED_IN_COUNTRY = 24;

    // ─────────────────────────────────────────────────────────────────────────
    // Index 25 — LAST_LOGGED_IN_IP_ADDRESS
    // ─────────────────────────────────────────────────────────────────────────
    /**
     * High-Cardinality IP | MD5(first 3 octets) % 10000 | normalised /10000
     * Source: {@code TokenizationRequest.lastLoggedInIpAddress}
     */
    public static final int LAST_LOGGED_IN_IP_ADDRESS = 25;
}
