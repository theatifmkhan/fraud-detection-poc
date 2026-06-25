package com.solarisbank.frauddetection.service;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.solarisbank.frauddetection.util.FeatureVector;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.Collections;

/**
 * Thread-safe singleton service that wraps the Microsoft ONNX Runtime for
 * local fraud probability inference.
 *
 * The {@link OrtSession} is shared across all concurrent requests — it is
 * thread-safe for inference per the ONNX Runtime specification.
 *
 * Input contract:
 *   - {@code float[]} of length {@link FeatureVector#FEATURE_COUNT} (10)
 *   - Produced by {@link EncodingService#encode(com.solarisbank.frauddetection.dto.TokenizationRequest)}
 *
 * Output:
 *   - Fraud probability score as {@code double} in range [0.0, 1.0]
 *   - Returns {@code -1.0} as a sentinel if inference fails (logged as ERROR)
 */
@Slf4j
@Service
public class ModelInferenceService {

    private static final double INFERENCE_ERROR_SENTINEL = -1.0;

    @Value("${fraud.onnx.input-node-name:float_input}")
    private String inputNodeName;

    @Value("${fraud.onnx.output-node-name:probabilities}")
    private String outputNodeName;

    private final OrtEnvironment ortEnvironment;
    private final OrtSession    ortSession;

    public ModelInferenceService(OrtEnvironment ortEnvironment, OrtSession ortSession) {
        this.ortEnvironment = ortEnvironment;
        this.ortSession     = ortSession;
    }

    /**
     * Runs inference on the provided feature vector and returns the fraud
     * probability score (class 1 probability from the Softmax output).
     *
     * @param features encoded feature array of length {@link FeatureVector#FEATURE_COUNT}
     * @return fraud probability in [0.0, 1.0], or {@code -1.0} on error
     */
    public double predictFraudScore(float[] features) {
        if (features == null || features.length != FeatureVector.FEATURE_COUNT) {
            log.error("ModelInferenceService: invalid feature vector — expected length {}, got {}",
                    FeatureVector.FEATURE_COUNT, features == null ? "null" : features.length);
            return INFERENCE_ERROR_SENTINEL;
        }

        OnnxTensor inputTensor = null;
        OrtSession.Result result = null;

        try {
            // Shape: [1, FEATURE_COUNT] — single sample batch
            float[][] input = new float[][]{features};
            inputTensor = OnnxTensor.createTensor(ortEnvironment, input);

            result = ortSession.run(
                    Collections.singletonMap(inputNodeName, inputTensor)
            );

            // Extract probabilities output: shape [1, 2]
            // Column 0 = P(legitimate), Column 1 = P(fraud)
            Object outputObj = result.get(outputNodeName)
                    .orElseThrow(() -> new OrtException(
                            "Output node '" + outputNodeName + "' not found in model. " +
                            "Check fraud.onnx.output-node-name in application.yml."
                    ))
                    .getValue();

            float[][] probabilities = (float[][]) outputObj;
            double fraudScore = probabilities[0][1];

            log.debug("ModelInferenceService: inference complete — score={}", fraudScore);
            return fraudScore;

        } catch (OrtException e) {
            log.error("ModelInferenceService: ONNX inference failed — tensor shape mismatch or model error: {}",
                    e.getMessage(), e);
            return INFERENCE_ERROR_SENTINEL;
        } finally {
            closeQuietly(inputTensor);
            closeQuietly(result);
        }
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private void closeQuietly(AutoCloseable closeable) {
        if (closeable != null) {
            try {
                closeable.close();
            } catch (Exception e) {
                log.warn("ModelInferenceService: error closing ONNX resource: {}", e.getMessage());
            }
        }
    }
}
