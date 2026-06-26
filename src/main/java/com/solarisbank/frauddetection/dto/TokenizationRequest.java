package com.solarisbank.frauddetection.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Request payload for the fraud evaluation endpoint.
 *
 * Fields mirror the Visa approveProvisioning API as declared in:
 * visa-tokenization/source/app/api/visa/approve_provisioning.rb
 *
 * Additional fields not present in the Visa API are sourced from:
 * - Decrypted encryptedData payload  (walletProvider* fields, risk scores)
 * - Cardholder address table         (nameOnAccount, cardholderCountry)
 * - MaxMind IP geolocation           (tokenProvisionIpCountry)
 * - User session / device details    (lastLoggedIn* fields)
 *
 * In production these additional fields are populated by upstream services
 * before the request reaches the fraud engine. For the POC they are accepted
 * directly in the request body; null/missing values fall back to defined
 * defaults in EncodingService.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class TokenizationRequest {

    // -------------------------------------------------------------------------
    // Path-level identifiers (from Visa API route params)
    // -------------------------------------------------------------------------

    /** Identifies the wallet provider (e.g. Apple Pay, Google Pay, Samsung Pay). */
    private String tokenRequestorId;

    /** Unique reference ID of the token being provisioned. */
    private String tokenReferenceId;

    // -------------------------------------------------------------------------
    // Required body fields (Visa approveProvisioning API)
    // -------------------------------------------------------------------------

    /**
     * Token metadata — contains assurance method, active/inactive/suspended
     * token counts and requestor details.
     */
    private TokenInfo tokenInfo;

    /** Reference ID of the PAN associated with this token request. */
    private String panReferenceId;

    /** Hashed email address of the wallet account holder. */
    private String walletAccountEmailAddressHash;

    /** Wallet provider's internal account identifier for the cardholder. */
    private String clientWalletAccountId;

    /**
     * Encrypted payload containing risk and cardholder verification data.
     * Not used directly by EncodingService — passed opaquely downstream.
     */
    private String encryptedData;

    // -------------------------------------------------------------------------
    // Optional body fields (Visa approveProvisioning API)
    // -------------------------------------------------------------------------

    /**
     * Source of the PAN in this tokenization request.
     * Expected values: KEY_ENTERED, ON_FILE, MOBILE_BANKING_APP, TOKEN,
     * CHIP_DIP, CONTACTLESS_TAP
     */
    private String panSource;

    /** Result code from the address verification check. */
    private String addressVerificationResultCode;

    /** Result code from the CVV2 verification. Expected: M, N, P, S, U */
    private String cvv2ResultsCode;

    /**
     * Mode in which the consumer entered their card details.
     * Expected values: KEY_ENTERED, CAMERA_CAPTURED, UNKNOWN
     */
    private String consumerEntryMode;

    /** Locale of the requesting client (e.g. "en-US", "de-DE"). */
    private String locale;

    /**
     * Device information — contains OS type, IP address, device type and
     * identifiers. Used by EncodingService to derive multiple features.
     */
    private DeviceInfo deviceInfo;

    /** Visa lifecycle trace ID for end-to-end request correlation. */
    private String lifeCycleTraceId;

    // -------------------------------------------------------------------------
    // Cardholder identity fields
    // Source: cardholder data / card record
    // -------------------------------------------------------------------------

    /**
     * Full name of the account holder as stored on the card record.
     * Feature: account_holder_name — Snowflake MD5 % 1000 buckets.
     * Default: null (encodes to bucket 0).
     */
    private String accountHolderName;

    /**
     * Name on account as sourced from the CARD_HOLDER_ADDRESSES table.
     * Treated as a separate feature from accountHolderName per training data schema.
     * Feature: NAME_ON_ACCOUNT — Snowflake MD5 % 1000 buckets.
     * Default: null (encodes to bucket 0).
     */
    private String nameOnAccount;

    /**
     * ISO 3166-1 alpha-2 country code of the cardholder's registered address.
     * Feature: CARDHOLDER_COUNTRY — Snowflake MD5 % 1000 buckets.
     * Default: null (encodes to bucket 0).
     */
    private String cardholderCountry;

    // -------------------------------------------------------------------------
    // Wallet provider risk signals
    // Source: decrypted from encryptedData.riskInformation in production
    // -------------------------------------------------------------------------

    /**
     * Wallet provider's assessment score for the requesting device.
     * Ordinal: 1 (low trust) to 5 (high trust). Default: 0.
     * Feature: wallet_provider_device_score — normalised by /5.
     */
    private Integer walletProviderDeviceScore;

    /**
     * Wallet provider's assessment score for the wallet account.
     * Ordinal: 1 (low trust) to 5 (high trust). Default: 0.
     * Feature: wallet_provider_account_score — normalised by /5.
     */
    private Integer walletProviderAccountScore;

    /**
     * Wallet provider's overall risk assessment for this provisioning event.
     * Ordinal: 0 (approve), 1 (decline), 2 (require step-up). Default: 0.
     * Feature: wallet_provider_risk_assessment — normalised by /2.
     */
    private Integer walletProviderRiskAssessment;

    /**
     * Wallet provider reason codes explaining the risk assessment.
     * Can be a single code (e.g. "01") or comma-separated (e.g. "01,A5,0G").
     * Feature: wallet_provider_reason_codes — Snowflake MD5 of full string % 100.
     * Default: null (encodes to bucket 0).
     */
    private String walletProviderReasonCodes;

    // -------------------------------------------------------------------------
    // Visa risk scores
    // Source: Visa token risk response
    // -------------------------------------------------------------------------

    /**
     * Combined risk assessment score from the Visa risk engine.
     * Ordinal: 1 to 5. Default: 0.
     * Feature: risk_assessment_score — normalised by /5.
     */
    private Integer riskAssessmentScore;

    /**
     * Visa's proprietary token risk score for this provisioning event.
     * Ordinal: 1 to 5. Default: 0.
     * Feature: visa_token_score — normalised by /5.
     */
    private Integer visaTokenScore;

    // -------------------------------------------------------------------------
    // IP geolocation
    // Source: MaxMind GeoIP lookup on deviceInfo.deviceIpAddressV4
    // -------------------------------------------------------------------------

    /**
     * ISO 3166-1 alpha-2 country code resolved from the provisioning device IP.
     * Feature: TOKEN_PROVISION_IP_COUNTRY — Snowflake MD5 % 10000 buckets.
     * Default: null (encodes to bucket 0).
     */
    private String tokenProvisionIpCountry;

    // -------------------------------------------------------------------------
    // Last-login session context
    // Source: user session store / DEVICE_DETAILS table
    // -------------------------------------------------------------------------

    /**
     * Device OS type used in the user's most recent login session.
     * Expected values: iOS, Android, Web. Default: null.
     * Feature: LAST_LOGGED_IN_DEVICE_TYPE — label encoded, normalised by /2.
     */
    private String lastLoggedInDeviceType;

    /**
     * Device name used in the user's most recent login session.
     * Feature: LAST_LOGGED_IN_DEVICE_NAME — Snowflake MD5 % 1000 buckets.
     * Default: null (encodes to bucket 0).
     */
    private String lastLoggedInDeviceName;

    /**
     * Country resolved from the IP used in the user's most recent login.
     * Source: DEVICE_DETAILS_DEVICE_IP_COUNTRY column.
     * Feature: LAST_LOGGED_IN_COUNTRY — Snowflake MD5 % 1000 buckets.
     * Default: null (encodes to bucket 0).
     */
    private String lastLoggedInCountry;

    /**
     * IPv4 address used in the user's most recent login session.
     * Feature: LAST_LOGGED_IN_IP_ADDRESS — MD5(first 3 octets) % 10000 buckets.
     * Default: null (encodes to bucket 0).
     */
    private String lastLoggedInIpAddress;
}
