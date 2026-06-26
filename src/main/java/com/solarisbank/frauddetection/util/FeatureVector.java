package com.solarisbank.frauddetection.util;

/**
 * Canonical feature vector definition for the fraud detection ONNX model.
 *
 * Defines the strict index order of all 53 features in the float[] array
 * passed to ModelInferenceService. This contract must match the column order
 * used during Snowflake data export and SageMaker model training exactly.
 *
 * Feature vector length: {@value FEATURE_COUNT}
 *
 * Source document: Shack.xlsx — "DataPoints Definition" sheet (26 data points).
 *
 * ── Encoding strategy by data type ──────────────────────────────────────────
 *
 *   High-Cardinality / Free Text  → Snowflake MD5 hash
 *     ABS(MD5_NUMBER_LOWER64(value)) % buckets, normalised by /buckets.
 *     Single float per field. Default 0.0f when null/blank.
 *
 *   IP Address                    → MD5 hash of first 3 octets, 10000 buckets.
 *     E.g. "192.168.1.55" → MD5("192.168.1") % 10000 / 10000.
 *
 *   Ordinal                       → raw integer / declared max.
 *     Default 0.0f when null.
 *
 *   Low-Cardinality (nominal)     → one-hot encoding.
 *     One binary float (1.0f or 0.0f) per category value.
 *     All zeros when value is null, blank, or unrecognised.
 *     Avoids the false ordinal relationship introduced by label encoding.
 *
 *   wallet_provider_reason_codes  → Snowflake MD5 hash % 100 (kept as single
 *     feature; multi-value field — one-hot expansion deferred for later).
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Index layout:
 *   0–13  High-cardinality / free-text hash features    (14 features)
 *  14–18  Ordinal features                              ( 5 features)
 *  19–20  One-hot: token_requestor_id                   ( 2 features)
 *  21–23  One-hot: consumer_entry_mode                  ( 3 features)
 *  24–32  One-hot: device_type                          ( 9 features)
 *  33–37  One-hot: token_type                           ( 5 features)
 *  38–43  One-hot: cvv2_results_code                    ( 6 features)
 *  44–49  One-hot: pan_source                           ( 6 features)
 *  50–52  One-hot: LAST_LOGGED_IN_DEVICE_TYPE           ( 3 features)
 * ─────────────────────────────────────────────────────────────────────────────
 */
public final class FeatureVector {

    private FeatureVector() {}

    /** Total number of features. Must match the ONNX model input shape [1, FEATURE_COUNT]. */
    public static final int FEATURE_COUNT = 53;

    // =========================================================================
    // HIGH-CARDINALITY / FREE-TEXT HASH FEATURES  (indices 0–13)
    // Encoding: Snowflake MD5 lower-64-bit hash % buckets, normalised by /buckets
    // =========================================================================

    /** Index 0  | account_holder_name    | MD5 % 1000 / 1000  | src: accountHolderName */
    public static final int ACCOUNT_HOLDER_NAME = 0;

    /** Index 1  | device_ip_address_v4   | MD5(3 octets) % 10000 / 10000 | src: deviceInfo.deviceIpAddressV4 */
    public static final int DEVICE_IP_ADDRESS_V4 = 1;

    /** Index 2  | device_name            | MD5 % 1000 / 1000  | src: deviceInfo.deviceName */
    public static final int DEVICE_NAME = 2;

    /** Index 3  | device_id              | MD5 % 1000 / 1000  | src: deviceInfo.deviceId */
    public static final int DEVICE_ID = 3;

    /** Index 4  | device_number          | MD5 % 1000 / 1000  | src: deviceInfo.deviceNumber */
    public static final int DEVICE_NUMBER = 4;

    /**
     * Index 5  | wallet_provider_reason_codes | MD5 of full string % 100 / 100
     * Multi-value field (e.g. "01,A5,0G") — kept as single hash feature for now.
     * src: walletProviderReasonCodes
     */
    public static final int WALLET_PROVIDER_REASON_CODES = 5;

    /** Index 6  | device_language_code   | MD5 % 1000 / 1000  | src: deviceInfo.deviceLanguageCode */
    public static final int DEVICE_LANGUAGE_CODE = 6;

    /** Index 7  | pan_reference_id       | MD5 % 1000 / 1000  | src: panReferenceId */
    public static final int PAN_REFERENCE_ID = 7;

    /** Index 8  | NAME_ON_ACCOUNT        | MD5 % 1000 / 1000  | src: nameOnAccount */
    public static final int NAME_ON_ACCOUNT = 8;

    /** Index 9  | CARDHOLDER_COUNTRY     | MD5 % 1000 / 1000  | src: cardholderCountry */
    public static final int CARDHOLDER_COUNTRY = 9;

    /** Index 10 | TOKEN_PROVISION_IP_COUNTRY | MD5 % 10000 / 10000 | src: tokenProvisionIpCountry */
    public static final int TOKEN_PROVISION_IP_COUNTRY = 10;

    /** Index 11 | LAST_LOGGED_IN_DEVICE_NAME | MD5 % 1000 / 1000  | src: lastLoggedInDeviceName */
    public static final int LAST_LOGGED_IN_DEVICE_NAME = 11;

    /** Index 12 | LAST_LOGGED_IN_COUNTRY | MD5 % 1000 / 1000  | src: lastLoggedInCountry */
    public static final int LAST_LOGGED_IN_COUNTRY = 12;

    /** Index 13 | LAST_LOGGED_IN_IP_ADDRESS | MD5(3 octets) % 10000 / 10000 | src: lastLoggedInIpAddress */
    public static final int LAST_LOGGED_IN_IP_ADDRESS = 13;

    // =========================================================================
    // ORDINAL FEATURES  (indices 14–18)
    // Encoding: raw integer / declared max. Default 0.0f when null.
    // =========================================================================

    /** Index 14 | wallet_provider_device_score    | Ordinal 1–5 / 5 | src: walletProviderDeviceScore */
    public static final int WALLET_PROVIDER_DEVICE_SCORE = 14;

    /** Index 15 | wallet_provider_account_score   | Ordinal 1–5 / 5 | src: walletProviderAccountScore */
    public static final int WALLET_PROVIDER_ACCOUNT_SCORE = 15;

    /** Index 16 | wallet_provider_risk_assessment | Ordinal 0–2 / 2 | src: walletProviderRiskAssessment */
    public static final int WALLET_PROVIDER_RISK_ASSESSMENT = 16;

    /** Index 17 | risk_assessment_score           | Ordinal 1–5 / 5 | src: riskAssessmentScore */
    public static final int RISK_ASSESSMENT_SCORE = 17;

    /** Index 18 | visa_token_score                | Ordinal 1–5 / 5 | src: visaTokenScore */
    public static final int VISA_TOKEN_SCORE = 18;

    // =========================================================================
    // ONE-HOT: token_requestor_id  (indices 19–20)
    // Values: APPLE, GOOGLE  — both 0 means OTHER / unknown
    // =========================================================================

    /** Index 19 | is_APPLE  | 1.0f if tokenRequestorId == "APPLE"  | src: tokenRequestorId */
    public static final int TOKEN_REQUESTOR_IS_APPLE = 19;

    /** Index 20 | is_GOOGLE | 1.0f if tokenRequestorId == "GOOGLE" | src: tokenRequestorId */
    public static final int TOKEN_REQUESTOR_IS_GOOGLE = 20;

    // =========================================================================
    // ONE-HOT: consumer_entry_mode  (indices 21–23)
    // Values: KEY_ENTERED, CAMERA_CAPTURED, UNKNOWN
    // =========================================================================

    /** Index 21 | is_KEY_ENTERED      | src: consumerEntryMode */
    public static final int CONSUMER_ENTRY_IS_KEY_ENTERED = 21;

    /** Index 22 | is_CAMERA_CAPTURED  | src: consumerEntryMode */
    public static final int CONSUMER_ENTRY_IS_CAMERA_CAPTURED = 22;

    /** Index 23 | is_UNKNOWN          | src: consumerEntryMode */
    public static final int CONSUMER_ENTRY_IS_UNKNOWN = 23;

    // =========================================================================
    // ONE-HOT: device_type  (indices 24–32)
    // Values: UNKNOWN, MOBILE_PHONE, TABLET, WATCH, MOBILEPHONE_OR_TABLET,
    //         PC, HOUSEHOLD_DEVICE, WEARABLE_DEVICE, AUTOMOBILE_DEVICE
    // =========================================================================

    /** Index 24 | is_UNKNOWN                | src: deviceInfo.deviceType */
    public static final int DEVICE_TYPE_IS_UNKNOWN = 24;

    /** Index 25 | is_MOBILE_PHONE           | src: deviceInfo.deviceType */
    public static final int DEVICE_TYPE_IS_MOBILE_PHONE = 25;

    /** Index 26 | is_TABLET                 | src: deviceInfo.deviceType */
    public static final int DEVICE_TYPE_IS_TABLET = 26;

    /** Index 27 | is_WATCH                  | src: deviceInfo.deviceType */
    public static final int DEVICE_TYPE_IS_WATCH = 27;

    /** Index 28 | is_MOBILEPHONE_OR_TABLET  | src: deviceInfo.deviceType */
    public static final int DEVICE_TYPE_IS_MOBILEPHONE_OR_TABLET = 28;

    /** Index 29 | is_PC                     | src: deviceInfo.deviceType */
    public static final int DEVICE_TYPE_IS_PC = 29;

    /** Index 30 | is_HOUSEHOLD_DEVICE       | src: deviceInfo.deviceType */
    public static final int DEVICE_TYPE_IS_HOUSEHOLD_DEVICE = 30;

    /** Index 31 | is_WEARABLE_DEVICE        | src: deviceInfo.deviceType */
    public static final int DEVICE_TYPE_IS_WEARABLE_DEVICE = 31;

    /** Index 32 | is_AUTOMOBILE_DEVICE      | src: deviceInfo.deviceType */
    public static final int DEVICE_TYPE_IS_AUTOMOBILE_DEVICE = 32;

    // =========================================================================
    // ONE-HOT: token_type  (indices 33–37)
    // Values: SECURE_ELEMENT, HCE, CARD_ON_FILE, ECOMMERCE, QRC
    // =========================================================================

    /** Index 33 | is_SECURE_ELEMENT  | src: tokenInfo.tokenType */
    public static final int TOKEN_TYPE_IS_SECURE_ELEMENT = 33;

    /** Index 34 | is_HCE             | src: tokenInfo.tokenType */
    public static final int TOKEN_TYPE_IS_HCE = 34;

    /** Index 35 | is_CARD_ON_FILE    | src: tokenInfo.tokenType */
    public static final int TOKEN_TYPE_IS_CARD_ON_FILE = 35;

    /** Index 36 | is_ECOMMERCE       | src: tokenInfo.tokenType */
    public static final int TOKEN_TYPE_IS_ECOMMERCE = 36;

    /** Index 37 | is_QRC             | src: tokenInfo.tokenType */
    public static final int TOKEN_TYPE_IS_QRC = 37;

    // =========================================================================
    // ONE-HOT: cvv2_results_code  (indices 38–43)
    // Values: M, N, P, S, U, NULL
    // =========================================================================

    /** Index 38 | is_M    — Match              | src: cvv2ResultsCode */
    public static final int CVV2_IS_M = 38;

    /** Index 39 | is_N    — No match           | src: cvv2ResultsCode */
    public static final int CVV2_IS_N = 39;

    /** Index 40 | is_P    — Not processed      | src: cvv2ResultsCode */
    public static final int CVV2_IS_P = 40;

    /** Index 41 | is_S    — Should be present  | src: cvv2ResultsCode */
    public static final int CVV2_IS_S = 41;

    /** Index 42 | is_U    — Unavailable        | src: cvv2ResultsCode */
    public static final int CVV2_IS_U = 42;

    /** Index 43 | is_NULL — Not provided       | src: cvv2ResultsCode */
    public static final int CVV2_IS_NULL = 43;

    // =========================================================================
    // ONE-HOT: pan_source  (indices 44–49)
    // Values: KEY_ENTERED, ON_FILE, MOBILE_BANKING_APP, TOKEN, CHIP_DIP, CONTACTLESS_TAP
    // =========================================================================

    /** Index 44 | is_KEY_ENTERED        | src: panSource */
    public static final int PAN_SOURCE_IS_KEY_ENTERED = 44;

    /** Index 45 | is_ON_FILE            | src: panSource */
    public static final int PAN_SOURCE_IS_ON_FILE = 45;

    /** Index 46 | is_MOBILE_BANKING_APP | src: panSource */
    public static final int PAN_SOURCE_IS_MOBILE_BANKING_APP = 46;

    /** Index 47 | is_TOKEN              | src: panSource */
    public static final int PAN_SOURCE_IS_TOKEN = 47;

    /** Index 48 | is_CHIP_DIP           | src: panSource */
    public static final int PAN_SOURCE_IS_CHIP_DIP = 48;

    /** Index 49 | is_CONTACTLESS_TAP    | src: panSource */
    public static final int PAN_SOURCE_IS_CONTACTLESS_TAP = 49;

    // =========================================================================
    // ONE-HOT: LAST_LOGGED_IN_DEVICE_TYPE  (indices 50–52)
    // Values: iOS, Android, Web
    // =========================================================================

    /** Index 50 | is_iOS     | src: lastLoggedInDeviceType */
    public static final int LAST_LOGIN_DEVICE_IS_IOS = 50;

    /** Index 51 | is_Android | src: lastLoggedInDeviceType */
    public static final int LAST_LOGIN_DEVICE_IS_ANDROID = 51;

    /** Index 52 | is_Web     | src: lastLoggedInDeviceType */
    public static final int LAST_LOGIN_DEVICE_IS_WEB = 52;
}
