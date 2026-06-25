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
 * Path parameters from the Visa API (tokenRequestorId, tokenReferenceId)
 * are included here as body fields for POC simplicity — no path param
 * routing is required on the frontend form.
 *
 * Required fields (as per Visa API):
 *   tokenRequestorId, tokenReferenceId, tokenInfo,
 *   panReferenceId, walletAccountEmailAddressHash,
 *   clientWalletAccountId, encryptedData
 *
 * Optional fields (as per Visa API):
 *   panSource, addressVerificationResultCode, cvv2ResultsCode,
 *   consumerEntryMode, locale, deviceInfo, lifeCycleTraceId
 *
 * All fields are nullable — EncodingService applies Snowflake-aligned
 * null-handling defaults before passing the feature vector to the ONNX model.
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
    // Required body fields
    // -------------------------------------------------------------------------

    /**
     * Token metadata object — contains assurance method, active/inactive/suspended
     * token counts, and requestor details.
     * Used by EncodingService to derive num_active_tokens and num_suspended_tokens features.
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
    // Optional body fields
    // -------------------------------------------------------------------------

    /**
     * Source of the PAN in this tokenization request.
     * Example values: "MOBILE_BANKING_APP", "MANUAL_ENTRY", "ON_FILE".
     */
    private String panSource;

    /** Result code from the address verification check. */
    private String addressVerificationResultCode;

    /** Result code from the CVV2 verification. */
    private String cvv2ResultsCode;

    /**
     * Mode in which the consumer entered their card details.
     * Example values: "MANUAL", "ON_FILE".
     */
    private String consumerEntryMode;

    /** Locale of the requesting client (e.g. "en-US", "de-DE"). */
    private String locale;

    /**
     * Device information object — contains OS type, IP address, device type and identifiers.
     * Used by EncodingService to derive device_os, ip_first_octet, and ip_is_private features.
     */
    private DeviceInfo deviceInfo;

    /** Visa lifecycle trace ID for end-to-end request correlation. */
    private String lifeCycleTraceId;
}
