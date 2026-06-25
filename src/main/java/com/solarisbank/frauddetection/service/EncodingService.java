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
import java.time.DayOfWeek;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;

/**
 * Converts a raw {@link TokenizationRequest} into a fixed-length float array
 * ready for ONNX model inference.
 *
 * Feature order is defined by {@link FeatureVector} constants and must match
 * the column order used during Snowflake data export and model training exactly.
 *
 * Encoding rules and null-handling defaults are loaded from
 * {@code classpath:mappings/encoding-mappings.json} at startup.
 */
@Slf4j
@Service
public class EncodingService {

    private static final String MAPPINGS_PATH = "mappings/encoding-mappings.json";

    // Normalisation caps (loaded from mappings JSON)
    private int activeTokensMax;
    private int suspendedTokensMax;
    private int tokenRefIdLenMax;
    private int walletAccountHashMod;

    // Default fallback values (loaded from mappings JSON)
    private float defaultHourOfDay;
    private float defaultDayOfWeek;
    private float defaultIpFirstOctet;
    private float defaultIpIsPrivate;
    private float defaultActiveTokens;
    private float defaultSuspendedTokens;
    private float defaultWalletHash;
    private float defaultTokenRefLen;

    private final ObjectMapper objectMapper;

    public EncodingService(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    @PostConstruct
    void loadMappings() {
        try (InputStream is = new ClassPathResource(MAPPINGS_PATH).getInputStream()) {
            JsonNode root = objectMapper.readTree(is);

            JsonNode caps = root.get("normalisationCaps");
            activeTokensMax    = caps.get("activeTokensMax").asInt();
            suspendedTokensMax = caps.get("suspendedTokensMax").asInt();
            tokenRefIdLenMax   = caps.get("tokenRefIdLenMax").asInt();
            walletAccountHashMod = caps.get("walletAccountHashMod").asInt();

            JsonNode defaults = root.get("defaults");
            defaultHourOfDay      = (float) defaults.get("hourOfDay").asDouble();
            defaultDayOfWeek      = (float) defaults.get("dayOfWeek").asDouble();
            defaultIpFirstOctet   = (float) defaults.get("ipFirstOctet").asDouble();
            defaultIpIsPrivate    = (float) defaults.get("ipIsPrivate").asDouble();
            defaultActiveTokens   = (float) defaults.get("activeTokens").asDouble();
            defaultSuspendedTokens = (float) defaults.get("suspendedTokens").asDouble();
            defaultWalletHash     = (float) defaults.get("walletHash").asDouble();
            defaultTokenRefLen    = (float) defaults.get("tokenRefLen").asDouble();

            log.info("EncodingService: mappings loaded successfully from {}", MAPPINGS_PATH);
        } catch (Exception e) {
            throw new IllegalStateException("Failed to load encoding mappings from " + MAPPINGS_PATH, e);
        }
    }

    /**
     * Encodes a {@link TokenizationRequest} into a {@code float[]} of length
     * {@link FeatureVector#FEATURE_COUNT} suitable for ONNX model inference.
     *
     * @param request the raw tokenization request (fields may be null)
     * @return encoded feature vector; never null
     */
    public float[] encode(TokenizationRequest request) {
        float[] features = new float[FeatureVector.FEATURE_COUNT];

        encodeDeviceOs(features, request.getDeviceInfo());
        encodeTemporalFeatures(features);
        encodeIpFeatures(features, request.getDeviceInfo());
        encodeTokenCountFeatures(features, request.getTokenInfo());
        encodeIdentityFeatures(features, request);

        return features;
    }

    // -------------------------------------------------------------------------
    // Feature encoders
    // -------------------------------------------------------------------------

    /**
     * One-hot encodes deviceInfo.osType into indices 0 (iOS) and 1 (Android).
     * Any value other than "iOS" or "Android" (case-insensitive) encodes as [0, 0].
     */
    private void encodeDeviceOs(float[] features, DeviceInfo deviceInfo) {
        if (deviceInfo == null || deviceInfo.getOsType() == null) {
            features[FeatureVector.DEVICE_OS_IOS]     = 0.0f;
            features[FeatureVector.DEVICE_OS_ANDROID] = 0.0f;
            log.warn("EncodingService: deviceInfo.osType is null — encoding as [0, 0]");
            return;
        }

        String os = deviceInfo.getOsType().trim();
        if (os.equalsIgnoreCase("iOS")) {
            features[FeatureVector.DEVICE_OS_IOS]     = 1.0f;
            features[FeatureVector.DEVICE_OS_ANDROID] = 0.0f;
        } else if (os.equalsIgnoreCase("Android")) {
            features[FeatureVector.DEVICE_OS_IOS]     = 0.0f;
            features[FeatureVector.DEVICE_OS_ANDROID] = 1.0f;
        } else {
            features[FeatureVector.DEVICE_OS_IOS]     = 0.0f;
            features[FeatureVector.DEVICE_OS_ANDROID] = 0.0f;
            log.warn("EncodingService: unrecognised osType '{}' — encoding as [0, 0]", os);
        }
    }

    /**
     * Extracts hour of day (UTC) and day of week from the server-side ingestion
     * time. Uses the current UTC clock — no timestamp field from the request
     * to avoid timezone drift between client and server.
     * Defaults to -1.0f on any error.
     */
    private void encodeTemporalFeatures(float[] features) {
        try {
            ZonedDateTime now = ZonedDateTime.now(ZoneOffset.UTC);
            features[FeatureVector.HOUR_OF_DAY] = (float) now.getHour();
            features[FeatureVector.DAY_OF_WEEK] = (float) now.getDayOfWeek().getValue(); // 1=Mon, 7=Sun
        } catch (Exception e) {
            log.warn("EncodingService: failed to extract temporal features — applying defaults", e);
            features[FeatureVector.HOUR_OF_DAY] = defaultHourOfDay;
            features[FeatureVector.DAY_OF_WEEK] = defaultDayOfWeek;
        }
    }

    /**
     * Derives ip_first_octet and ip_is_private from deviceInfo.deviceIpAddressV4.
     * RFC-1918 private ranges: 10.x.x.x, 172.16–31.x.x, 192.168.x.x.
     * Defaults to 0.0f for both if IP is null or malformed.
     */
    private void encodeIpFeatures(float[] features, DeviceInfo deviceInfo) {
        if (deviceInfo == null || isBlank(deviceInfo.getDeviceIpAddressV4())) {
            features[FeatureVector.IP_FIRST_OCTET] = defaultIpFirstOctet;
            features[FeatureVector.IP_IS_PRIVATE]  = defaultIpIsPrivate;
            log.warn("EncodingService: deviceIpAddressV4 is null/blank — applying defaults");
            return;
        }

        String ip = deviceInfo.getDeviceIpAddressV4().trim();
        String[] octets = ip.split("\\.");

        if (octets.length != 4) {
            features[FeatureVector.IP_FIRST_OCTET] = defaultIpFirstOctet;
            features[FeatureVector.IP_IS_PRIVATE]  = defaultIpIsPrivate;
            log.warn("EncodingService: malformed IPv4 '{}' — applying defaults", ip);
            return;
        }

        try {
            int firstOctet  = Integer.parseInt(octets[0]);
            int secondOctet = Integer.parseInt(octets[1]);

            features[FeatureVector.IP_FIRST_OCTET] = (float) firstOctet;
            features[FeatureVector.IP_IS_PRIVATE]  = isPrivateIp(firstOctet, secondOctet) ? 1.0f : 0.0f;
        } catch (NumberFormatException e) {
            features[FeatureVector.IP_FIRST_OCTET] = defaultIpFirstOctet;
            features[FeatureVector.IP_IS_PRIVATE]  = defaultIpIsPrivate;
            log.warn("EncodingService: could not parse IP octets from '{}' — applying defaults", ip);
        }
    }

    /**
     * Normalises active and suspended token counts from tokenInfo.
     * Formula: min(count, cap) / cap. Defaults to 0.0f if tokenInfo or field is null.
     */
    private void encodeTokenCountFeatures(float[] features, TokenInfo tokenInfo) {
        if (tokenInfo == null) {
            features[FeatureVector.NUM_ACTIVE_TOKENS_NORM]    = defaultActiveTokens;
            features[FeatureVector.NUM_SUSPENDED_TOKENS_NORM] = defaultSuspendedTokens;
            log.warn("EncodingService: tokenInfo is null — applying token count defaults");
            return;
        }

        features[FeatureVector.NUM_ACTIVE_TOKENS_NORM] = tokenInfo.getNumberOfActiveTokensForPAN() != null
                ? Math.min(tokenInfo.getNumberOfActiveTokensForPAN(), activeTokensMax) / (float) activeTokensMax
                : defaultActiveTokens;

        features[FeatureVector.NUM_SUSPENDED_TOKENS_NORM] = tokenInfo.getNumberOfSuspendedTokensForPAN() != null
                ? Math.min(tokenInfo.getNumberOfSuspendedTokensForPAN(), suspendedTokensMax) / (float) suspendedTokensMax
                : defaultSuspendedTokens;
    }

    /**
     * Encodes clientWalletAccountId hash and tokenReferenceId length.
     * Defaults to 0.0f if the respective field is null or blank.
     */
    private void encodeIdentityFeatures(float[] features, TokenizationRequest request) {
        // Client wallet account hash norm
        if (isBlank(request.getClientWalletAccountId())) {
            features[FeatureVector.CLIENT_WALLET_ACCOUNT_HASH_NORM] = defaultWalletHash;
            log.warn("EncodingService: clientWalletAccountId is null/blank — applying default");
        } else {
            int hash = Math.abs(request.getClientWalletAccountId().hashCode()) % walletAccountHashMod;
            features[FeatureVector.CLIENT_WALLET_ACCOUNT_HASH_NORM] = hash / (float) walletAccountHashMod;
        }

        // Token reference ID length norm
        if (isBlank(request.getTokenReferenceId())) {
            features[FeatureVector.TOKEN_REFERENCE_ID_LEN_NORM] = defaultTokenRefLen;
            log.warn("EncodingService: tokenReferenceId is null/blank — applying default");
        } else {
            int len = Math.min(request.getTokenReferenceId().length(), tokenRefIdLenMax);
            features[FeatureVector.TOKEN_REFERENCE_ID_LEN_NORM] = len / (float) tokenRefIdLenMax;
        }
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private boolean isPrivateIp(int first, int second) {
        return first == 10
                || first == 192 && second == 168
                || first == 172 && second >= 16 && second <= 31;
    }

    private boolean isBlank(String s) {
        return s == null || s.isBlank();
    }
}
