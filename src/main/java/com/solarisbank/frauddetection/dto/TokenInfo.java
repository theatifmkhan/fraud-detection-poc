package com.solarisbank.frauddetection.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Represents the tokenInfo object from the Visa approveProvisioning API.
 * All fields are optional — EncodingService applies null-safe defaults.
 *
 * Source: visa-tokenization/source/app/api/visa/approve_provisioning.rb
 * Canonical field list from: spec/unit/.../token_provisioning_request_spec.rb
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class TokenInfo {

    private String tokenAssuranceMethod;
    private Integer numberOfActiveTokensForPAN;
    private Integer numberOfInactiveTokensForPAN;
    private Integer numberOfSuspendedTokensForPAN;
    private String originalTokenReferenceId;
    private String originalTokenType;
    private String originalTokenAssuranceMethod;
    private String tokenRequestorName;
    private String tokenRequestorType;

    /** Present in integration test fixtures (approve_provisioning_spec.rb). */
    private String tokenType;
}
