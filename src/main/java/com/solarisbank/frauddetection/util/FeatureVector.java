package com.solarisbank.frauddetection.util;

/**
 * Canonical feature vector definition for the fraud detection ONNX model.
 *
 * Defines the strict index order of all 53 features in the float[] array
 * passed to ModelInferenceService.
 *
 * Feature vector length: {@value FEATURE_COUNT}
 *
 * ── IMPORTANT: Column ordering contract ──────────────────────────────────────
 * The index order MUST match the column order in the Snowflake training data
 * export, which follows the DataPoints Definition sheet in Shack.xlsx row by row.
 * One-hot expanded columns for a given field appear immediately after their
 * source row's position in the sheet.
 *
 * See also: Shack.xlsx → "Model Training Columns" sheet for the exact
 * Snowflake SQL expressions and training column names.
 *
 * ── Encoding strategy by data type ──────────────────────────────────────────
 *   High-Cardinality / Free Text  → Snowflake MD5 hash
 *     ABS(MD5_NUMBER_LOWER64(value)) % buckets, normalised by /buckets.
 *     Default 0.0f when null/blank.
 *
 *   IP Address                    → MD5 hash of first 3 octets, 10000 buckets.
 *
 *   Ordinal                       → raw integer / declared max. Default 0.0f.
 *
 *   Low-Cardinality (nominal)     → one-hot encoding.
 *     One binary float (1.0f or 0.0f) per category value.
 *     All zeros when null, blank, or unrecognised.
 *
 *   wallet_provider_reason_codes  → MD5 hash % 100 (multi-value; deferred).
 * ─────────────────────────────────────────────────────────────────────────────
 */
public final class FeatureVector {

    private FeatureVector() {}

    /** Total features. Must match ONNX model input shape [1, FEATURE_COUNT]. */
    public static final int FEATURE_COUNT = 53;

    // ── Row 1: account_holder_name ───────────────────────────────────────────
    /** Index 0  | account_holder_name | MD5 % 1000 / 1000 | src: accountHolderName */
    public static final int ACCOUNT_HOLDER_NAME = 0;

    // ── Row 2: token_requestor_id (one-hot) ──────────────────────────────────
    /** Index 1  | token_requestor_is_apple  | src: tokenRequestorId */
    public static final int TOKEN_REQUESTOR_IS_APPLE = 1;

    /** Index 2  | token_requestor_is_google | src: tokenRequestorId */
    public static final int TOKEN_REQUESTOR_IS_GOOGLE = 2;

    // ── Row 3: consumer_entry_mode (one-hot) ──────────────────────────────────
    /** Index 3  | consumer_entry_is_key_entered     | src: consumerEntryMode */
    public static final int CONSUMER_ENTRY_IS_KEY_ENTERED = 3;

    /** Index 4  | consumer_entry_is_camera_captured | src: consumerEntryMode */
    public static final int CONSUMER_ENTRY_IS_CAMERA_CAPTURED = 4;

    /** Index 5  | consumer_entry_is_unknown         | src: consumerEntryMode */
    public static final int CONSUMER_ENTRY_IS_UNKNOWN = 5;

    // ── Row 4: device_ip_address_v4 ───────────────────────────────────────────
    /** Index 6  | device_ip_address_v4 | MD5(3 octets) % 10000 / 10000 | src: deviceInfo.deviceIpAddressV4 */
    public static final int DEVICE_IP_ADDRESS_V4 = 6;

    // ── Row 5: wallet_provider_device_score ───────────────────────────────────
    /** Index 7  | wallet_provider_device_score | Ordinal 1–5 / 5 | src: walletProviderDeviceScore */
    public static final int WALLET_PROVIDER_DEVICE_SCORE = 7;

    // ── Row 6: device_type (one-hot, 9 values) ────────────────────────────────
    /** Index 8  | device_type_is_unknown               | src: deviceInfo.deviceType */
    public static final int DEVICE_TYPE_IS_UNKNOWN = 8;

    /** Index 9  | device_type_is_mobile_phone          | src: deviceInfo.deviceType */
    public static final int DEVICE_TYPE_IS_MOBILE_PHONE = 9;

    /** Index 10 | device_type_is_tablet                | src: deviceInfo.deviceType */
    public static final int DEVICE_TYPE_IS_TABLET = 10;

    /** Index 11 | device_type_is_watch                 | src: deviceInfo.deviceType */
    public static final int DEVICE_TYPE_IS_WATCH = 11;

    /** Index 12 | device_type_is_mobilephone_or_tablet | src: deviceInfo.deviceType */
    public static final int DEVICE_TYPE_IS_MOBILEPHONE_OR_TABLET = 12;

    /** Index 13 | device_type_is_pc                    | src: deviceInfo.deviceType */
    public static final int DEVICE_TYPE_IS_PC = 13;

    /** Index 14 | device_type_is_household_device      | src: deviceInfo.deviceType */
    public static final int DEVICE_TYPE_IS_HOUSEHOLD_DEVICE = 14;

    /** Index 15 | device_type_is_wearable_device       | src: deviceInfo.deviceType */
    public static final int DEVICE_TYPE_IS_WEARABLE_DEVICE = 15;

    /** Index 16 | device_type_is_automobile_device     | src: deviceInfo.deviceType */
    public static final int DEVICE_TYPE_IS_AUTOMOBILE_DEVICE = 16;

    // ── Row 7: wallet_provider_account_score ──────────────────────────────────
    /** Index 17 | wallet_provider_account_score | Ordinal 1–5 / 5 | src: walletProviderAccountScore */
    public static final int WALLET_PROVIDER_ACCOUNT_SCORE = 17;

    // ── Row 8: device_name ────────────────────────────────────────────────────
    /** Index 18 | device_name | MD5 % 1000 / 1000 | src: deviceInfo.deviceName */
    public static final int DEVICE_NAME = 18;

    // ── Row 9: device_id ──────────────────────────────────────────────────────
    /** Index 19 | device_id | MD5 % 1000 / 1000 | src: deviceInfo.deviceId */
    public static final int DEVICE_ID = 19;

    // ── Row 10: wallet_provider_risk_assessment ───────────────────────────────
    /** Index 20 | wallet_provider_risk_assessment | Ordinal 0–2 / 2 | src: walletProviderRiskAssessment */
    public static final int WALLET_PROVIDER_RISK_ASSESSMENT = 20;

    // ── Row 11: device_number ─────────────────────────────────────────────────
    /** Index 21 | device_number | MD5 % 1000 / 1000 | src: deviceInfo.deviceNumber */
    public static final int DEVICE_NUMBER = 21;

    // ── Row 12: wallet_provider_reason_codes ──────────────────────────────────
    /** Index 22 | wallet_provider_reason_codes | MD5 % 100 / 100 (multi-value, deferred) | src: walletProviderReasonCodes */
    public static final int WALLET_PROVIDER_REASON_CODES = 22;

    // ── Row 13: token_type (one-hot, 5 values) ────────────────────────────────
    /** Index 23 | token_type_is_secure_element | src: tokenInfo.tokenType */
    public static final int TOKEN_TYPE_IS_SECURE_ELEMENT = 23;

    /** Index 24 | token_type_is_hce            | src: tokenInfo.tokenType */
    public static final int TOKEN_TYPE_IS_HCE = 24;

    /** Index 25 | token_type_is_card_on_file   | src: tokenInfo.tokenType */
    public static final int TOKEN_TYPE_IS_CARD_ON_FILE = 25;

    /** Index 26 | token_type_is_ecommerce      | src: tokenInfo.tokenType */
    public static final int TOKEN_TYPE_IS_ECOMMERCE = 26;

    /** Index 27 | token_type_is_qrc            | src: tokenInfo.tokenType */
    public static final int TOKEN_TYPE_IS_QRC = 27;

    // ── Row 14: risk_assessment_score ─────────────────────────────────────────
    /** Index 28 | risk_assessment_score | Ordinal 1–5 / 5 | src: riskAssessmentScore */
    public static final int RISK_ASSESSMENT_SCORE = 28;

    // ── Row 15: device_language_code ──────────────────────────────────────────
    /** Index 29 | device_language_code | MD5 % 1000 / 1000 | src: deviceInfo.deviceLanguageCode */
    public static final int DEVICE_LANGUAGE_CODE = 29;

    // ── Row 16: pan_reference_id ──────────────────────────────────────────────
    /** Index 30 | pan_reference_id | MD5 % 1000 / 1000 | src: panReferenceId */
    public static final int PAN_REFERENCE_ID = 30;

    // ── Row 17: cvv2_results_code (one-hot, 6 values) ─────────────────────────
    /** Index 31 | cvv2_is_m    — Match             | src: cvv2ResultsCode */
    public static final int CVV2_IS_M = 31;

    /** Index 32 | cvv2_is_n    — No match          | src: cvv2ResultsCode */
    public static final int CVV2_IS_N = 32;

    /** Index 33 | cvv2_is_p    — Not processed     | src: cvv2ResultsCode */
    public static final int CVV2_IS_P = 33;

    /** Index 34 | cvv2_is_s    — Should be present | src: cvv2ResultsCode */
    public static final int CVV2_IS_S = 34;

    /** Index 35 | cvv2_is_u    — Unavailable       | src: cvv2ResultsCode */
    public static final int CVV2_IS_U = 35;

    /** Index 36 | cvv2_is_null — Not provided      | src: cvv2ResultsCode */
    public static final int CVV2_IS_NULL = 36;

    // ── Row 18: pan_source (one-hot, 6 values) ────────────────────────────────
    /** Index 37 | pan_source_is_key_entered        | src: panSource */
    public static final int PAN_SOURCE_IS_KEY_ENTERED = 37;

    /** Index 38 | pan_source_is_on_file            | src: panSource */
    public static final int PAN_SOURCE_IS_ON_FILE = 38;

    /** Index 39 | pan_source_is_mobile_banking_app | src: panSource */
    public static final int PAN_SOURCE_IS_MOBILE_BANKING_APP = 39;

    /** Index 40 | pan_source_is_token              | src: panSource */
    public static final int PAN_SOURCE_IS_TOKEN = 40;

    /** Index 41 | pan_source_is_chip_dip           | src: panSource */
    public static final int PAN_SOURCE_IS_CHIP_DIP = 41;

    /** Index 42 | pan_source_is_contactless_tap    | src: panSource */
    public static final int PAN_SOURCE_IS_CONTACTLESS_TAP = 42;

    // ── Row 19: visa_token_score ──────────────────────────────────────────────
    /** Index 43 | visa_token_score | Ordinal 1–5 / 5 | src: visaTokenScore */
    public static final int VISA_TOKEN_SCORE = 43;

    // ── Row 20: NAME_ON_ACCOUNT ───────────────────────────────────────────────
    /** Index 44 | name_on_account | MD5 % 1000 / 1000 | src: nameOnAccount */
    public static final int NAME_ON_ACCOUNT = 44;

    // ── Row 21: CARDHOLDER_COUNTRY ────────────────────────────────────────────
    /** Index 45 | cardholder_country | MD5 % 1000 / 1000 | src: cardholderCountry */
    public static final int CARDHOLDER_COUNTRY = 45;

    // ── Row 22: TOKEN_PROVISION_IP_COUNTRY ────────────────────────────────────
    /** Index 46 | token_provision_ip_country | MD5 % 10000 / 10000 | src: tokenProvisionIpCountry */
    public static final int TOKEN_PROVISION_IP_COUNTRY = 46;

    // ── Row 23: LAST_LOGGED_IN_DEVICE_TYPE (one-hot, 3 values) ───────────────
    /** Index 47 | last_login_device_is_ios     | src: lastLoggedInDeviceType */
    public static final int LAST_LOGIN_DEVICE_IS_IOS = 47;

    /** Index 48 | last_login_device_is_android | src: lastLoggedInDeviceType */
    public static final int LAST_LOGIN_DEVICE_IS_ANDROID = 48;

    /** Index 49 | last_login_device_is_web     | src: lastLoggedInDeviceType */
    public static final int LAST_LOGIN_DEVICE_IS_WEB = 49;

    // ── Row 24: LAST_LOGGED_IN_DEVICE_NAME ───────────────────────────────────
    /** Index 50 | last_logged_in_device_name | MD5 % 1000 / 1000 | src: lastLoggedInDeviceName */
    public static final int LAST_LOGGED_IN_DEVICE_NAME = 50;

    // ── Row 25: LAST_LOGGED_IN_COUNTRY ────────────────────────────────────────
    /** Index 51 | last_logged_in_country | MD5 % 1000 / 1000 | src: lastLoggedInCountry */
    public static final int LAST_LOGGED_IN_COUNTRY = 51;

    // ── Row 26: LAST_LOGGED_IN_IP_ADDRESS ─────────────────────────────────────
    /** Index 52 | last_logged_in_ip_address | MD5(3 octets) % 10000 / 10000 | src: lastLoggedInIpAddress */
    public static final int LAST_LOGGED_IN_IP_ADDRESS = 52;
}
