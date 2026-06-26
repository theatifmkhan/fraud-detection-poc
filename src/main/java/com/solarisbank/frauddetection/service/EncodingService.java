package com.solarisbank.frauddetection.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.solarisbank.frauddetection.dto.DeviceInfo;
import com.solarisbank.frauddetection.dto.TokenInfo;
import com.solarisbank.frauddetection.dto.TokenizationRequest;
import com.solarisbank.frauddetection.util.FeatureVector;
import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;

import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/**
 * Converts a raw {@link TokenizationRequest} into a fixed-length float array
 * ready for ONNX model inference.
 *
 * Feature order follows the DataPoints Definition sheet in Shack.xlsx row by row,
 * with one-hot expanded columns placed immediately after their source row position.
 * This order is the strict contract with the Snowflake training data export.
 *
 * Encoding strategies:
 *   - High-cardinality / free-text: Snowflake-compatible MD5 hash
 *       ABS(MD5_NUMBER_LOWER64(value)) % buckets / buckets
 *   - IP address: MD5 of first 3 octets, 10000 buckets
 *   - Ordinal: raw integer / declared max
 *   - Low-cardinality nominal: one-hot (1.0f for matching category, 0.0f otherwise)
 */
@Slf4j
@Service
public class EncodingService {

    private static final String MAPPINGS_PATH = "mappings/encoding-mappings.json";

    // Hash bucket sizes
    private long bucketsAccountHolderName;
    private long bucketsIpAddress;
    private long bucketsDeviceName;
    private long bucketsDeviceId;
    private long bucketsDeviceNumber;
    private long bucketsWalletReasonCodes;
    private long bucketsDeviceLanguageCode;
    private long bucketsPanReferenceId;
    private long bucketsNameOnAccount;
    private long bucketsCardholderCountry;
    private long bucketsTokenProvisionIpCountry;
    private long bucketsLastLoggedInDeviceName;
    private long bucketsLastLoggedInCountry;

    // Ordinal max values
    private int maxWalletProviderDeviceScore;
    private int maxWalletProviderAccountScore;
    private int maxWalletProviderRiskAssessment;
    private int maxRiskAssessmentScore;
    private int maxVisaTokenScore;

    private final ObjectMapper objectMapper;

    public EncodingService(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    @PostConstruct
    void loadMappings() {
        try (InputStream is = new ClassPathResource(MAPPINGS_PATH).getInputStream()) {
            JsonNode root = objectMapper.readTree(is);

            JsonNode hb = root.get("hashBuckets");
            bucketsAccountHolderName      = hb.get("accountHolderName").asLong();
            bucketsIpAddress              = hb.get("ipAddress").asLong();
            bucketsDeviceName             = hb.get("deviceName").asLong();
            bucketsDeviceId               = hb.get("deviceId").asLong();
            bucketsDeviceNumber           = hb.get("deviceNumber").asLong();
            bucketsWalletReasonCodes      = hb.get("walletProviderReasonCodes").asLong();
            bucketsDeviceLanguageCode     = hb.get("deviceLanguageCode").asLong();
            bucketsPanReferenceId         = hb.get("panReferenceId").asLong();
            bucketsNameOnAccount          = hb.get("nameOnAccount").asLong();
            bucketsCardholderCountry      = hb.get("cardholderCountry").asLong();
            bucketsTokenProvisionIpCountry = hb.get("tokenProvisionIpCountry").asLong();
            bucketsLastLoggedInDeviceName = hb.get("lastLoggedInDeviceName").asLong();
            bucketsLastLoggedInCountry    = hb.get("lastLoggedInCountry").asLong();

            JsonNode om = root.get("ordinalMax");
            maxWalletProviderDeviceScore    = om.get("walletProviderDeviceScore").asInt();
            maxWalletProviderAccountScore   = om.get("walletProviderAccountScore").asInt();
            maxWalletProviderRiskAssessment = om.get("walletProviderRiskAssessment").asInt();
            maxRiskAssessmentScore          = om.get("riskAssessmentScore").asInt();
            maxVisaTokenScore               = om.get("visaTokenScore").asInt();

            log.info("EncodingService: mappings loaded from {}", MAPPINGS_PATH);
        } catch (Exception e) {
            throw new IllegalStateException("Failed to load encoding mappings from " + MAPPINGS_PATH, e);
        }
    }

    /**
     * Encodes a {@link TokenizationRequest} into a {@code float[]} of length
     * {@link FeatureVector#FEATURE_COUNT} (53) suitable for ONNX model inference.
     *
     * Order mirrors the DataPoints Definition sheet in Shack.xlsx row by row.
     */
    public float[] encode(TokenizationRequest request) {
        float[] f = new float[FeatureVector.FEATURE_COUNT];
        DeviceInfo d = request.getDeviceInfo();
        TokenInfo  t = request.getTokenInfo();

        // ── Row 1: account_holder_name (index 0) ─────────────────────────────
        f[FeatureVector.ACCOUNT_HOLDER_NAME] =
                hashNorm(request.getAccountHolderName(), bucketsAccountHolderName);

        // ── Row 2: token_requestor_id → one-hot (indices 1–2) ────────────────
        oneHot(f, request.getTokenRequestorId(),
               new String[]{"APPLE", "GOOGLE"},
               new int[]{FeatureVector.TOKEN_REQUESTOR_IS_APPLE,
                         FeatureVector.TOKEN_REQUESTOR_IS_GOOGLE},
               "tokenRequestorId");

        // ── Row 3: consumer_entry_mode → one-hot (indices 3–5) ───────────────
        oneHot(f, request.getConsumerEntryMode(),
               new String[]{"KEY_ENTERED", "CAMERA_CAPTURED", "UNKNOWN"},
               new int[]{FeatureVector.CONSUMER_ENTRY_IS_KEY_ENTERED,
                         FeatureVector.CONSUMER_ENTRY_IS_CAMERA_CAPTURED,
                         FeatureVector.CONSUMER_ENTRY_IS_UNKNOWN},
               "consumerEntryMode");

        // ── Row 4: device_ip_address_v4 (index 6) ────────────────────────────
        f[FeatureVector.DEVICE_IP_ADDRESS_V4] =
                ipHashNorm(d != null ? d.getDeviceIpAddressV4() : null,
                           bucketsIpAddress, "deviceIpAddressV4");

        // ── Row 5: wallet_provider_device_score (index 7) ────────────────────
        f[FeatureVector.WALLET_PROVIDER_DEVICE_SCORE] =
                ordinalNorm(request.getWalletProviderDeviceScore(),
                            maxWalletProviderDeviceScore, "walletProviderDeviceScore");

        // ── Row 6: device_type → one-hot (indices 8–16) ──────────────────────
        oneHot(f, d != null ? d.getDeviceType() : null,
               new String[]{"UNKNOWN", "MOBILE_PHONE", "TABLET", "WATCH",
                             "MOBILEPHONE_OR_TABLET", "PC", "HOUSEHOLD_DEVICE",
                             "WEARABLE_DEVICE", "AUTOMOBILE_DEVICE"},
               new int[]{FeatureVector.DEVICE_TYPE_IS_UNKNOWN,
                         FeatureVector.DEVICE_TYPE_IS_MOBILE_PHONE,
                         FeatureVector.DEVICE_TYPE_IS_TABLET,
                         FeatureVector.DEVICE_TYPE_IS_WATCH,
                         FeatureVector.DEVICE_TYPE_IS_MOBILEPHONE_OR_TABLET,
                         FeatureVector.DEVICE_TYPE_IS_PC,
                         FeatureVector.DEVICE_TYPE_IS_HOUSEHOLD_DEVICE,
                         FeatureVector.DEVICE_TYPE_IS_WEARABLE_DEVICE,
                         FeatureVector.DEVICE_TYPE_IS_AUTOMOBILE_DEVICE},
               "deviceType");

        // ── Row 7: wallet_provider_account_score (index 17) ──────────────────
        f[FeatureVector.WALLET_PROVIDER_ACCOUNT_SCORE] =
                ordinalNorm(request.getWalletProviderAccountScore(),
                            maxWalletProviderAccountScore, "walletProviderAccountScore");

        // ── Row 8: device_name (index 18) ────────────────────────────────────
        f[FeatureVector.DEVICE_NAME] =
                hashNorm(d != null ? d.getDeviceName() : null, bucketsDeviceName);

        // ── Row 9: device_id (index 19) ──────────────────────────────────────
        f[FeatureVector.DEVICE_ID] =
                hashNorm(d != null ? d.getDeviceId() : null, bucketsDeviceId);

        // ── Row 10: wallet_provider_risk_assessment (index 20) ───────────────
        f[FeatureVector.WALLET_PROVIDER_RISK_ASSESSMENT] =
                ordinalNorm(request.getWalletProviderRiskAssessment(),
                            maxWalletProviderRiskAssessment, "walletProviderRiskAssessment");

        // ── Row 11: device_number (index 21) ─────────────────────────────────
        f[FeatureVector.DEVICE_NUMBER] =
                hashNorm(d != null ? d.getDeviceNumber() : null, bucketsDeviceNumber);

        // ── Row 12: wallet_provider_reason_codes (index 22) ──────────────────
        f[FeatureVector.WALLET_PROVIDER_REASON_CODES] =
                hashNorm(request.getWalletProviderReasonCodes(), bucketsWalletReasonCodes);

        // ── Row 13: token_type → one-hot (indices 23–27) ─────────────────────
        oneHot(f, t != null ? t.getTokenType() : null,
               new String[]{"SECURE_ELEMENT", "HCE", "CARD_ON_FILE", "ECOMMERCE", "QRC"},
               new int[]{FeatureVector.TOKEN_TYPE_IS_SECURE_ELEMENT,
                         FeatureVector.TOKEN_TYPE_IS_HCE,
                         FeatureVector.TOKEN_TYPE_IS_CARD_ON_FILE,
                         FeatureVector.TOKEN_TYPE_IS_ECOMMERCE,
                         FeatureVector.TOKEN_TYPE_IS_QRC},
               "tokenType");

        // ── Row 14: risk_assessment_score (index 28) ─────────────────────────
        f[FeatureVector.RISK_ASSESSMENT_SCORE] =
                ordinalNorm(request.getRiskAssessmentScore(),
                            maxRiskAssessmentScore, "riskAssessmentScore");

        // ── Row 15: device_language_code (index 29) ──────────────────────────
        f[FeatureVector.DEVICE_LANGUAGE_CODE] =
                hashNorm(d != null ? d.getDeviceLanguageCode() : null, bucketsDeviceLanguageCode);

        // ── Row 16: pan_reference_id (index 30) ──────────────────────────────
        f[FeatureVector.PAN_REFERENCE_ID] =
                hashNorm(request.getPanReferenceId(), bucketsPanReferenceId);

        // ── Row 17: cvv2_results_code → one-hot (indices 31–36) ──────────────
        oneHot(f, request.getCvv2ResultsCode(),
               new String[]{"M", "N", "P", "S", "U", "NULL"},
               new int[]{FeatureVector.CVV2_IS_M, FeatureVector.CVV2_IS_N,
                         FeatureVector.CVV2_IS_P, FeatureVector.CVV2_IS_S,
                         FeatureVector.CVV2_IS_U, FeatureVector.CVV2_IS_NULL},
               "cvv2ResultsCode");

        // ── Row 18: pan_source → one-hot (indices 37–42) ─────────────────────
        oneHot(f, request.getPanSource(),
               new String[]{"KEY_ENTERED", "ON_FILE", "MOBILE_BANKING_APP",
                             "TOKEN", "CHIP_DIP", "CONTACTLESS_TAP"},
               new int[]{FeatureVector.PAN_SOURCE_IS_KEY_ENTERED,
                         FeatureVector.PAN_SOURCE_IS_ON_FILE,
                         FeatureVector.PAN_SOURCE_IS_MOBILE_BANKING_APP,
                         FeatureVector.PAN_SOURCE_IS_TOKEN,
                         FeatureVector.PAN_SOURCE_IS_CHIP_DIP,
                         FeatureVector.PAN_SOURCE_IS_CONTACTLESS_TAP},
               "panSource");

        // ── Row 19: visa_token_score (index 43) ──────────────────────────────
        f[FeatureVector.VISA_TOKEN_SCORE] =
                ordinalNorm(request.getVisaTokenScore(), maxVisaTokenScore, "visaTokenScore");

        // ── Row 20: NAME_ON_ACCOUNT (index 44) ───────────────────────────────
        f[FeatureVector.NAME_ON_ACCOUNT] =
                hashNorm(request.getNameOnAccount(), bucketsNameOnAccount);

        // ── Row 21: CARDHOLDER_COUNTRY (index 45) ────────────────────────────
        f[FeatureVector.CARDHOLDER_COUNTRY] =
                hashNorm(request.getCardholderCountry(), bucketsCardholderCountry);

        // ── Row 22: TOKEN_PROVISION_IP_COUNTRY (index 46) ────────────────────
        f[FeatureVector.TOKEN_PROVISION_IP_COUNTRY] =
                hashNorm(request.getTokenProvisionIpCountry(), bucketsTokenProvisionIpCountry);

        // ── Row 23: LAST_LOGGED_IN_DEVICE_TYPE → one-hot (indices 47–49) ─────
        oneHot(f, request.getLastLoggedInDeviceType(),
               new String[]{"iOS", "Android", "Web"},
               new int[]{FeatureVector.LAST_LOGIN_DEVICE_IS_IOS,
                         FeatureVector.LAST_LOGIN_DEVICE_IS_ANDROID,
                         FeatureVector.LAST_LOGIN_DEVICE_IS_WEB},
               "lastLoggedInDeviceType");

        // ── Row 24: LAST_LOGGED_IN_DEVICE_NAME (index 50) ────────────────────
        f[FeatureVector.LAST_LOGGED_IN_DEVICE_NAME] =
                hashNorm(request.getLastLoggedInDeviceName(), bucketsLastLoggedInDeviceName);

        // ── Row 25: LAST_LOGGED_IN_COUNTRY (index 51) ────────────────────────
        f[FeatureVector.LAST_LOGGED_IN_COUNTRY] =
                hashNorm(request.getLastLoggedInCountry(), bucketsLastLoggedInCountry);

        // ── Row 26: LAST_LOGGED_IN_IP_ADDRESS (index 52) ─────────────────────
        f[FeatureVector.LAST_LOGGED_IN_IP_ADDRESS] =
                ipHashNorm(request.getLastLoggedInIpAddress(),
                           bucketsIpAddress, "lastLoggedInIpAddress");

        return f;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Snowflake-compatible MD5 hash  ABS(MD5_NUMBER_LOWER64(value)) % buckets
    // ─────────────────────────────────────────────────────────────────────────

    long snowflakeHash(String input, long buckets) {
        if (isBlank(input)) return 0L;
        try {
            MessageDigest md = MessageDigest.getInstance("MD5");
            byte[] digest = md.digest(input.getBytes(StandardCharsets.UTF_8));
            long lower64 = ByteBuffer.wrap(digest, 8, 8).getLong();
            return Math.abs(lower64) % buckets;
        } catch (NoSuchAlgorithmException e) {
            log.error("EncodingService: MD5 not available — returning 0", e);
            return 0L;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Encoding helpers
    // ─────────────────────────────────────────────────────────────────────────

    private float hashNorm(String value, long buckets) {
        return snowflakeHash(value, buckets) / (float) buckets;
    }

    private float ipHashNorm(String ip, long buckets, String fieldName) {
        if (isBlank(ip)) return 0.0f;
        String[] parts = ip.trim().split("\\.");
        if (parts.length < 3) {
            log.warn("EncodingService: malformed IPv4 '{}' for {} — default 0.0", ip, fieldName);
            return 0.0f;
        }
        String first3 = parts[0] + "." + parts[1] + "." + parts[2];
        return snowflakeHash(first3, buckets) / (float) buckets;
    }

    private float ordinalNorm(Integer value, int max, String fieldName) {
        if (value == null) return 0.0f;
        return value / (float) max;
    }

    private void oneHot(float[] f, String value, String[] categories,
                        int[] indices, String fieldName) {
        if (isBlank(value)) return;
        String v = value.trim();
        for (int i = 0; i < categories.length; i++) {
            if (categories[i].equalsIgnoreCase(v)) {
                f[indices[i]] = 1.0f;
                return;
            }
        }
        log.warn("EncodingService: unrecognised value '{}' for {} — all zeros", value, fieldName);
    }

    private boolean isBlank(String s) {
        return s == null || s.isBlank();
    }
}
