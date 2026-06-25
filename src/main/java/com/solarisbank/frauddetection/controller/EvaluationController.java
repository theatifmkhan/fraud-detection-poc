package com.solarisbank.frauddetection.controller;

import com.solarisbank.frauddetection.dto.EvaluationResponse;
import com.solarisbank.frauddetection.dto.TokenizationRequest;
import com.solarisbank.frauddetection.service.BedrockExplainerService;
import com.solarisbank.frauddetection.service.EncodingService;
import com.solarisbank.frauddetection.service.ModelInferenceService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * REST controller that orchestrates the dual-path fraud evaluation pipeline.
 *
 * Endpoint: POST /api/v1/evaluate
 *
 * Pipeline:
 *   1. EncodingService   — encodes raw request into float[10] feature vector
 *   2. ModelInferenceService — runs ONNX inference, returns fraud probability score
 *   3. Decision logic    — maps score to APPROVED / REVIEW / REJECTED
 *   4. BedrockExplainerService — triggered synchronously for REJECTED decisions only,
 *      generates a natural-language Analyst Audit Report via Claude 3.5 Sonnet
 *   5. Returns EvaluationResponse with score, decision, and optional aiExplanation
 *
 * Decision thresholds are configurable via application.yml:
 *   fraud.threshold.review  (default: 0.40) — score >= this -> REVIEW
 *   fraud.threshold.reject  (default: 0.65) — score >= this -> REJECTED + Bedrock explanation
 */
@Slf4j
@RestController
@RequestMapping("/api/v1")
public class EvaluationController {

    private static final String DECISION_APPROVED = "APPROVED";
    private static final String DECISION_REVIEW   = "REVIEW";
    private static final String DECISION_REJECTED = "REJECTED";

    @Value("${fraud.threshold.review:0.40}")
    private double reviewThreshold;

    @Value("${fraud.threshold.reject:0.65}")
    private double rejectThreshold;

    private final EncodingService         encodingService;
    private final ModelInferenceService   modelInferenceService;
    private final BedrockExplainerService bedrockExplainerService;

    public EvaluationController(EncodingService encodingService,
                                ModelInferenceService modelInferenceService,
                                BedrockExplainerService bedrockExplainerService) {
        this.encodingService         = encodingService;
        this.modelInferenceService   = modelInferenceService;
        this.bedrockExplainerService = bedrockExplainerService;
    }

    /**
     * Evaluates a tokenization request for fraud risk.
     *
     * @param request raw tokenization request payload
     * @return evaluation result containing score, decision, and optional AI explanation
     */
    @PostMapping("/evaluate")
    public ResponseEntity<EvaluationResponse> evaluate(@RequestBody TokenizationRequest request) {
        log.info("EvaluationController: received request — tokenReferenceId={}, tokenRequestorId={}",
                request.getTokenReferenceId(), request.getTokenRequestorId());

        // Step 1: encode features
        float[] features = encodingService.encode(request);
        log.debug("EvaluationController: features encoded -> length={}", features.length);

        // Step 2: run ONNX inference
        double score = modelInferenceService.predictFraudScore(features);
        log.info("EvaluationController: fraud score={}", score);

        // Step 3: determine decision
        String decision = resolveDecision(score);
        log.info("EvaluationController: decision={}", decision);

        // Step 4: conditionally trigger Bedrock explanation
        String aiExplanation = null;
        if (DECISION_REJECTED.equals(decision)) {
            log.info("EvaluationController: score >= reject threshold ({}) — triggering Bedrock explanation", rejectThreshold);
            aiExplanation = bedrockExplainerService.explain(request, score);
        }

        // Step 5: build and return response
        EvaluationResponse response = EvaluationResponse.builder()
                .score(score)
                .decision(decision)
                .aiExplanation(aiExplanation)
                .build();

        log.info("EvaluationController: response built — score={}, decision={}, hasExplanation={}",
                score, decision, aiExplanation != null);

        return ResponseEntity.ok(response);
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /**
     * Maps a fraud score to a decision string based on configured thresholds.
     *
     * Special case: score of -1.0 (inference error sentinel) is treated as REVIEW
     * and must be escalated to the on-call engineer per fraud policy.
     *
     * @param score fraud probability from ModelInferenceService
     * @return "APPROVED", "REVIEW", or "REJECTED"
     */
    private String resolveDecision(double score) {
        if (score < 0) {
            log.warn("EvaluationController: inference error sentinel score={} — defaulting to REVIEW", score);
            return DECISION_REVIEW;
        }
        if (score >= rejectThreshold) return DECISION_REJECTED;
        if (score >= reviewThreshold) return DECISION_REVIEW;
        return DECISION_APPROVED;
    }
}
