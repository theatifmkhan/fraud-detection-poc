package com.solarisbank.frauddetection.config;

import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;

/**
 * Ensures the AWS region is available to the AWS SDK's
 * DefaultAwsRegionProviderChain at the earliest possible point in
 * the Spring lifecycle.
 *
 * Problem: the AWS SDK resolves the region via DefaultAwsRegionProviderChain
 * (env vars → system props → config file → EC2 metadata) before Spring AI's
 * BedrockConnectionConfiguration has had a chance to create its Region @Bean.
 * This causes a startup failure if AWS_REGION / AWS_DEFAULT_REGION is not
 * set as an environment variable.
 *
 * Fix: read spring.ai.bedrock.aws.region from application.yml and write it
 * into the aws.region system property immediately, making it visible to the
 * SDK's provider chain for any subsequent AWS client instantiation.
 */
@Slf4j
@Configuration
public class BedrockRegionConfig {

    @Value("${spring.ai.bedrock.aws.region:eu-central-1}")
    private String region;

    @PostConstruct
    public void setAwsRegionSystemProperty() {
        System.setProperty("aws.region", region);
        log.info("BedrockRegionConfig: aws.region system property set to '{}'", region);
    }
}
