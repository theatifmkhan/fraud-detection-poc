package com.solarisbank.frauddetection.service;

import com.solarisbank.frauddetection.dto.DeviceInfo;
import com.solarisbank.frauddetection.dto.TokenInfo;
import com.solarisbank.frauddetection.dto.TokenizationRequest;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.stereotype.Service;

/**
 * Generates a natural-language fraud analysis explanation using Amazon Bedrock
 * (Claude 3.5 Sonnet via Spring AI ChatClient + Bedrock Converse API).
 *
 * Called synchronously by the EvaluationController only when the fraud score
 * meets or exceeds the configured reject threshold (default: 0.65).
 *
 * The explanation is structured as an "Analyst Audit Report" and combines:
 * - The raw tokenization request profile
 * - The ML fraud probability score
 * - Relevant internal security policy context from DocumentIngestionService
 *
 * Response latency is typically 1–3 seconds depending on Bedrock throughput.
 */
@Slf4j
@Service
public class BedrockExplainerService {

    private static final String SYSTEM_PROMPT = """
            You are a Fraud Analyst Copilot for a card tokenization platform.
            Your role is to analyse flagged tokenization provisioning requests and produce
            concise, factual audit reports for the fraud operations team.

            When given a request profile, an ML fraud score, and relevant security policy
            context, you must:
            1. Identify the 2–4 most significant risk signals present in the request
            2. Reference the applicable policy rules that were triggered
            3. Explain the likely fraud pattern these signals suggest
            4. Keep your response to 4–6 sentences maximum — be precise and actionable

            Do not speculate beyond the data provided. Do not repeat the raw input fields verbatim.
            Write in plain English suitable for a fraud analyst, not a developer.
            """;

    private static final String USER_PROMPT_TEMPLATE = """
            ## Fraud Evaluation Request

            **ML Fraud Score:** {score} (threshold for rejection: 0.65)

            **Request Profile:**
            - Token Requestor ID: {tokenRequestorId}
            - Token Reference ID: {tokenReferenceId}
            - Client Wallet Account ID: {clientWalletAccountId}
            - PAN Reference ID: {panReferenceId}
            - PAN Source: {panSource}
            - Consumer Entry Mode: {consumerEntryMode}
            - Locale: {locale}
            - Device OS Type: {osType}
            - Device Type: {deviceType}
            - Device IP Address: {ipAddress}
            - Active Tokens for PAN: {activeTokens}
            - Suspended Tokens for PAN: {suspendedTokens}
            - Inactive Tokens for PAN: {inactiveTokens}

            **Internal Security Policy Context:**
            {policyContext}

            Based on the above, produce a concise Analyst Audit Report explaining why this
            request was flagged as high-risk fraud.
            """;

    private final ChatClient chatClient;
    private final DocumentIngestionService documentIngestionService;

    public BedrockExplainerService(ChatClient.Builder chatClientBuilder,
                                   DocumentIngestionService documentIngestionService) {
        this.chatClient = chatClientBuilder
                .defaultSystem(SYSTEM_PROMPT)
                .build();
        this.documentIngestionService = documentIngestionService;
    }

    /**
     * Generates a fraud explanation for the given request and score.
     *
     * @param request   the original tokenization request
     * @param score     the fraud probability score from ModelInferenceService
     * @return          natural-language audit report string, or an error message if Bedrock call fails
     */
    public String explain(TokenizationRequest request, double score) {
        log.info("BedrockExplainerService: generating explanation for score={}", score);

        try {
            String prompt = buildUserPrompt(request, score);

            String explanation = chatClient
                    .prompt()
                    .user(prompt)
                    .call()
                    .content();

            log.info("BedrockExplainerService: explanation generated ({} chars)", explanation.length());
            return explanation;

        } catch (Exception e) {
            log.error("BedrockExplainerService: Bedrock call failed: {}", e.getMessage(), e);
            return "AI explanation unavailable — Bedrock call failed. Please review the request manually. Error: " + e.getMessage();
        }
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private String buildUserPrompt(TokenizationRequest request, double score) {
        DeviceInfo device   = request.getDeviceInfo();
        TokenInfo   token   = request.getTokenInfo();
        String policyContext = documentIngestionService.getPolicyContext();

        return USER_PROMPT_TEMPLATE
                .replace("{score}",            String.format("%.4f", score))
                .replace("{tokenRequestorId}", safeStr(request.getTokenRequestorId()))
                .replace("{tokenReferenceId}", safeStr(request.getTokenReferenceId()))
                .replace("{clientWalletAccountId}", safeStr(request.getClientWalletAccountId()))
                .replace("{panReferenceId}",    safeStr(request.getPanReferenceId()))
                .replace("{panSource}",         safeStr(request.getPanSource()))
                .replace("{consumerEntryMode}", safeStr(request.getConsumerEntryMode()))
                .replace("{locale}",            safeStr(request.getLocale()))
                .replace("{osType}",            device != null ? safeStr(device.getOsType())           : "null")
                .replace("{deviceType}",        device != null ? safeStr(device.getDeviceType())       : "null")
                .replace("{ipAddress}",         device != null ? safeStr(device.getDeviceIpAddressV4()) : "null")
                .replace("{activeTokens}",      token  != null && token.getNumberOfActiveTokensForPAN()    != null
                        ? String.valueOf(token.getNumberOfActiveTokensForPAN())    : "null")
                .replace("{suspendedTokens}",   token  != null && token.getNumberOfSuspendedTokensForPAN() != null
                        ? String.valueOf(token.getNumberOfSuspendedTokensForPAN()) : "null")
                .replace("{inactiveTokens}",    token  != null && token.getNumberOfInactiveTokensForPAN()  != null
                        ? String.valueOf(token.getNumberOfInactiveTokensForPAN())  : "null")
                .replace("{policyContext}",     policyContext);
    }

    private String safeStr(String value) {
        return value != null && !value.isBlank() ? value : "null";
    }
}
