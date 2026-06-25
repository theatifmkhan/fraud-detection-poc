package com.solarisbank.frauddetection.config;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;

import java.io.IOException;
import java.io.InputStream;

/**
 * Spring configuration for the ONNX Runtime.
 *
 * Creates a single {@link OrtEnvironment} and {@link OrtSession} as application-scoped
 * singletons. Both are thread-safe for concurrent inference calls.
 *
 * The model file is loaded from the path specified by {@code fraud.onnx.model-path}
 * in application.yml (default: {@code classpath:models/fraud_model.onnx}).
 *
 * Startup fails fast with a clear exception if the model file is missing or malformed —
 * this prevents silent failures where inference would throw NPEs at request time.
 */
@Slf4j
@Configuration
public class OnnxConfig {

    @Value("${fraud.onnx.model-path:classpath:models/fraud_model.onnx}")
    private String modelPath;

    private final ResourceLoader resourceLoader;

    public OnnxConfig(ResourceLoader resourceLoader) {
        this.resourceLoader = resourceLoader;
    }

    /**
     * Global ONNX Runtime environment. One per JVM process.
     */
    @Bean(destroyMethod = "close")
    public OrtEnvironment ortEnvironment() {
        log.info("OnnxConfig: initialising OrtEnvironment");
        return OrtEnvironment.getEnvironment();
    }

    /**
     * ONNX inference session loaded from {@code fraud.onnx.model-path}.
     * Fails fast at startup if the model file cannot be found or parsed.
     */
    @Bean(destroyMethod = "close")
    public OrtSession ortSession(OrtEnvironment ortEnvironment) throws OrtException, IOException {
        Resource resource = resourceLoader.getResource(modelPath);

        if (!resource.exists()) {
            throw new IllegalStateException(
                    "ONNX model file not found at: " + modelPath +
                    ". Run scripts/generate_dummy_onnx.py to create a placeholder model."
            );
        }

        try (InputStream is = resource.getInputStream()) {
            byte[] modelBytes = is.readAllBytes();
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            OrtSession session = ortEnvironment.createSession(modelBytes, options);
            log.info("OnnxConfig: OrtSession loaded from {} ({} bytes)", modelPath, modelBytes.length);
            log.info("OnnxConfig: model inputs  -> {}", session.getInputNames());
            log.info("OnnxConfig: model outputs -> {}", session.getOutputNames());
            return session;
        }
    }
}
