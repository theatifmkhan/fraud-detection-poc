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
import java.util.HashMap;
import java.util.Map;

/**
 * Converts a raw {@link TokenizationRequest} into a fixed-length float array
 * ready for ONNX model inference.
 *
 * Feature order is defined by {@link FeatureVector} constants (26 features)
 * and must match the column order used during Snowflake data export and model
 * training exactly.
 *
 * Hashing strategy: all High-Cardinality and Free-Text fields use the
 * Snowflake-compatible MD5 hash:
 *   ABS(MD5_NUMBER_LOWER64(value)) % buckets
 * which replicates Snowflake's built-in MD5_NUMBER_LOWER64 function, ensuring
 * that training-time (Snowflake) and serving-time (Java) bucket assignments
 * are identical.
 *
 * Encoding rules and bucket sizes are loaded from
 * {@code classpath:mappings/encoding-mappings.json} at startup.
 */
@Slf4j
@Service
public class EncodingService {

    private static final String MAPPINGS_PATH = "mappings/encoding-mappings.json";

    // Low-cardinality label maps loaded from JSON
    private Map<String, Integer> tokenRequestorIdMap;
    private Map<String, Integer> consumerEntryModeMap;
    private Map<String, Integer> deviceTypeMap;
    private Map<String, Integer> tokenTypeMap;
    private Map<String, Integer> cvv2ResultsCodeMap;
    private Map<String, Integer> panSourceMap;
    private Map<String, Integer> lastLoggedInDeviceTypeMap;

    // Normalisation denominators for low-cardinality
    private int tokenRequestorIdMax;
    private int consumerEntryModeMax;
    private int deviceTypeMax;
    private int tokenTypeMax;
    private int cvv2ResultsCodeMax;
    private int panSourceMax;
    private int lastLoggedInDeviceTypeMax;

    // Hash bucket sizes
    private long bucketsAccountHolderName;
    private long bucketsDeviceName;
    private long bucketsDeviceId;
    private long bucketsDeviceNumber;
    private long bucketsWalletReasonCodes;
    private long bucketsDeviceLanguageCode;
    private long bucketsPanReferenceId;
    private long bucketsNameOnAccount;
    private long bucketsCardholderCountry;
    private long bucketsLastLoggedInDeviceName;
    private long bucketsLastLoggedInCountry;
    private long bucketsIpAddress;
    private long bucketsTokenProvisionIpCountry;

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

            JsonNode lc = root.get("lowCardinality");
            tokenRequestorIdMap       = toMap(lc.get("tokenRequestorId"));
            tokenRequestorIdMax       = lc.get("tokenRequestorIdMax").asInt();
            consumerEntryModeMap      = toMap(lc.get("consumerEntryMode"));
            consumerEntryModeMax      = lc.get("consumerEntryModeMax").asInt();
            deviceTypeMap             = toMap(lc.get("deviceType"));
            deviceTypeMax             = lc.get("deviceTypeMax").asInt();
            tokenTypeMap              = toMap(lc.get("tokenType"));
            tokenTypeMax              = lc.get("tokenTypeMax").asInt();
            cvv2ResultsCodeMap        = toMap(lc.get("cvv2ResultsCode"));
            cvv2ResultsCodeMax        = lc.get("cvv2ResultsCodeMax").asInt();
            panSourceMap              = toMap(lc.get("panSource"));
            panSourceMax              = lc.get("panSourceMax").asInt();
            lastLoggedInDeviceTypeMap = toMap(lc.get("lastLoggedInDeviceType"));
            lastLoggedInDeviceTypeMax = lc.get("lastLoggedInDeviceTypeMax").asInt();

            JsonNode hb = root.get("hashBuckets");
            bucketsAccountHolderName      = hb.get("accountHolderName").asLong();
            bucketsDeviceName             = hb.get("deviceName").asLong();
            bucketsDeviceId               = hb.get("deviceId").asLong();
            bucketsDeviceNumber           = hb.get("deviceNumber").asLong();
            bucketsWalletReasonCodes      = hb.get("walletProviderReasonCodes").asLong();
            bucketsDeviceLanguageCode     = hb.get("deviceLanguageCode").asLong();
            bucketsPanReferenceId         = hb.get("panReferenceId").asLong();
            bucketsNameOnAccount          = hb.get("nameOnAccount").asLong();
            bucketsCardholderCountry      = hb.get("cardholderCountry").asLong();
            bucketsLastLoggedInDeviceName = hb.get("lastLoggedInDeviceName").asLong();
            bucketsLastLoggedInCountry    = hb.get("lastLoggedInCountry").asLong();
            bucketsIpAddress              = hb.get("ipAddress").asLong();
            bucketsTokenProvisionIpCountry = hb.get("tokenProvisionIpCountry").asLong();

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
     * {@link FeatureVector#FEATURE_COUNT} (26) suitable for ONNX model inference.
     *
     * @param request the raw tokenization request (fields may be null)
     * @return encoded feature vector; never null
     */
    public float[] encode(TokenizationRequest request) {
        float[] f = new float[FeatureVector.FEATURE_COUNT];
        DeviceInfo d = request.getDeviceInfo();
        TokenInfo  t = request.getTokenInfo();

        // 0 — account_holder_name
        f[FeatureVector.ACCOUNT_HOLDER_NAME] =
                hashNorm(request.getAccountHolderName(), bucketsAccountHolderName);

        // 1 — token_requestor_id
        f[FeatureVector.TOKEN_REQUESTOR_ID] =
                labelNorm(request.getTokenRequestorId(), tokenRequestorIdMap, tokenRequestorIdMax,
                          "tokenRequestorId (default OTHER=2)", 2);

        // 2 — consumer_entry_mode
        f[FeatureVector.CONSUMER_ENTRY_MODE] =
                labelNorm(request.getConsumerEntryMode(), consumerEntryModeMap, consumerEntryModeMax,
                          "consumerEntryMode");

        // 3 — device_ip_address_v4
        f[FeatureVector.DEVICE_IP_ADDRESS_V4] =
                ipHashNorm(d != null ? d.getDeviceIpAddressV4() : null, bucketsIpAddress, "deviceIpAddressV4");

        // 4 — wallet_provider_device_score
        f[FeatureVector.WALLET_PROVIDER_DEVICE_SCORE] =
                ordinalNorm(request.getWalletProviderDeviceScore(), maxWalletProviderDeviceScore,
                            "walletProviderDeviceScore");

        // 5 — device_type
        f[FeatureVector.DEVICE_TYPE] =
                labelNorm(d != null ? d.getDeviceType() : null, deviceTypeMap, deviceTypeMax, "deviceType");

        // 6 — wallet_provider_account_score
        f[FeatureVector.WALLET_PROVIDER_ACCOUNT_SCORE] =
                ordinalNorm(request.getWalletProviderAccountScore(), maxWalletProviderAccountScore,
                            "walletProviderAccountScore");

        // 7 — device_name
        f[FeatureVector.DEVICE_NAME] =
                hashNorm(d != null ? d.getDeviceName() : null, bucketsDeviceName);

        // 8 — device_id
        f[FeatureVector.DEVICE_ID] =
                hashNorm(d != null ? d.getDeviceId() : null, bucketsDeviceId);

        // 9 — wallet_provider_risk_assessment
        f[FeatureVector.WALLET_PROVIDER_RISK_ASSESSMENT] =
                ordinalNorm(request.getWalletProviderRiskAssessment(), maxWalletProviderRiskAssessment,
                            "walletProviderRiskAssessment");

        // 10 — device_number
        f[FeatureVector.DEVICE_NUMBER] =
                hashNorm(d != null ? d.getDeviceNumber() : null, bucketsDeviceNumber);

        // 11 — wallet_provider_reason_codes (hash full string, handles multi-value)
        f[FeatureVector.WALLET_PROVIDER_REASON_CODES] =
                hashNorm(request.getWalletProviderReasonCodes(), bucketsWalletReasonCodes);

        // 12 — token_type
        f[FeatureVector.TOKEN_TYPE] =
                labelNorm(t != null ? t.getTokenType() : null, tokenTypeMap, tokenTypeMax, "tokenType");

        // 13 — risk_assessment_score
        f[FeatureVector.RISK_ASSESSMENT_SCORE] =
                ordinalNorm(request.getRiskAssessmentScore(), maxRiskAssessmentScore, "riskAssessmentScore");

        // 14 — device_language_code
        f[FeatureVector.DEVICE_LANGUAGE_CODE] =
                hashNorm(d != null ? d.getDeviceLanguageCode() : null, bucketsDeviceLanguageCode);

        // 15 — pan_reference_id
        f[FeatureVector.PAN_REFERENCE_ID] =
                hashNorm(request.getPanReferenceId(), bucketsPanReferenceId);

        // 16 — cvv2_results_code
        f[FeatureVector.CVV2_RESULTS_CODE] =
                labelNorm(request.getCvv2ResultsCode(), cvv2ResultsCodeMap, cvv2ResultsCodeMax, "cvv2ResultsCode");

        // 17 — pan_source
        f[FeatureVector.PAN_SOURCE] =
                labelNorm(request.getPanSource(), panSourceMap, panSourceMax, "panSource");

        // 18 — visa_token_score
        f[FeatureVector.VISA_TOKEN_SCORE] =
                ordinalNorm(request.getVisaTokenScore(), maxVisaTokenScore, "visaTokenScore");

        // 19 — NAME_ON_ACCOUNT
        f[FeatureVector.NAME_ON_ACCOUNT] =
                hashNorm(request.getNameOnAccount(), bucketsNameOnAccount);

        // 20 — CARDHOLDER_COUNTRY
        f[FeatureVector.CARDHOLDER_COUNTRY] =
                hashNorm(request.getCardholderCountry(), bucketsCardholderCountry);

        // 21 — TOKEN_PROVISION_IP_COUNTRY
        f[FeatureVector.TOKEN_PROVISION_IP_COUNTRY] =
                hashNorm(request.getTokenProvisionIpCountry(), bucketsTokenProvisionIpCountry);

        // 22 — LAST_LOGGED_IN_DEVICE_TYPE
        f[FeatureVector.LAST_LOGGED_IN_DEVICE_TYPE] =
                labelNorm(request.getLastLoggedInDeviceType(), lastLoggedInDeviceTypeMap,
                          lastLoggedInDeviceTypeMax, "lastLoggedInDeviceType");

        // 23 — LAST_LOGGED_IN_DEVICE_NAME
        f[FeatureVector.LAST_LOGGED_IN_DEVICE_NAME] =
                hashNorm(request.getLastLoggedInDeviceName(), bucketsLastLoggedInDeviceName);

        // 24 — LAST_LOGGED_IN_COUNTRY
        f[FeatureVector.LAST_LOGGED_IN_COUNTRY] =
                hashNorm(request.getLastLoggedInCountry(), bucketsLastLoggedInCountry);

        // 25 — LAST_LOGGED_IN_IP_ADDRESS
        f[FeatureVector.LAST_LOGGED_IN_IP_ADDRESS] =
                ipHashNorm(request.getLastLoggedInIpAddress(), bucketsIpAddress, "lastLoggedInIpAddress");

        return f;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Snowflake-compatible MD5 hash (ABS(MD5_NUMBER_LOWER64(value)) % buckets)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Replicates Snowflake's {@code ABS(MD5_NUMBER_LOWER64(value)) % buckets}.
     *
     * <ol>
     *   <li>Compute MD5 digest of the UTF-8 encoded input string.</li>
     *   <li>Extract the lower 64 bits (last 8 bytes of the 16-byte digest).</li>
     *   <li>Return {@code Math.abs(lower64) % buckets}.</li>
     * </ol>
     *
     * @param input   the string to hash; null/blank returns 0
     * @param buckets modulus (number of hash buckets)
     * @return bucket index in [0, buckets)
     */
    long snowflakeHash(String input, long buckets) {
        if (isBlank(input)) return 0L;
        try {
            MessageDigest md = MessageDigest.getInstance("MD5");
            byte[] digest = md.digest(input.getBytes(StandardCharsets.UTF_8));
            // Lower 64 bits = last 8 bytes of the 16-byte MD5 array
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

    /** Hash a text field and normalise by the bucket count. */
    private float hashNorm(String value, long buckets) {
        return snowflakeHash(value, buckets) / (float) buckets;
    }

    /**
     * Hash the first 3 octets of an IPv4 address and normalise.
     * E.g. "192.168.1.55" → hash("192.168.1") % buckets / buckets.
     */
    private float ipHashNorm(String ip, long buckets, String fieldName) {
        if (isBlank(ip)) {
            log.warn("EncodingService: {} is null/blank — default 0.0", fieldName);
            return 0.0f;
        }
        String[] parts = ip.trim().split("\\.");
        if (parts.length < 3) {
            log.warn("EncodingService: malformed IPv4 '{}' for {} — default 0.0", ip, fieldName);
            return 0.0f;
        }
        String first3 = parts[0] + "." + parts[1] + "." + parts[2];
        return snowflakeHash(first3, buckets) / (float) buckets;
    }

    /** Look up a label in a map, normalise by max. Unknown values default to 0. */
    private float labelNorm(String value, Map<String, Integer> map, int max, String fieldName) {
        return labelNorm(value, map, max, fieldName, 0);
    }

    private float labelNorm(String value, Map<String, Integer> map, int max, String fieldName, int unknownLabel) {
        if (isBlank(value)) {
            log.warn("EncodingService: {} is null/blank — default {}", fieldName, unknownLabel / (float) max);
            return unknownLabel / (float) max;
        }
        Integer label = map.get(value.trim().toUpperCase());
        if (label == null) {
            // Try exact case match before giving up
            label = map.get(value.trim());
        }
        if (label == null) {
            log.warn("EncodingService: unknown value '{}' for {} — default {}", value, fieldName, unknownLabel / (float) max);
            return unknownLabel / (float) max;
        }
        return label / (float) max;
    }

    /** Normalise an ordinal integer by its declared max value. Null defaults to 0. */
    private float ordinalNorm(Integer value, int max, String fieldName) {
        if (value == null) {
            log.warn("EncodingService: {} is null — default 0.0", fieldName);
            return 0.0f;
        }
        return value / (float) max;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Utilities
    // ─────────────────────────────────────────────────────────────────────────

    private boolean isBlank(String s) {
        return s == null || s.isBlank();
    }

    private Map<String, Integer> toMap(JsonNode node) {
        Map<String, Integer> map = new HashMap<>();
        if (node != null) {
            node.fields().forEachRemaining(e -> map.put(e.getKey(), e.getValue().asInt()));
        }
        return map;
    }
}
