package com.solarisbank.frauddetection.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Loads internal fraud policy documents from {@code classpath:docs/} at application
 * startup and holds them in memory for use by {@link BedrockExplainerService}.
 *
 * Documents are loaded as plain text strings. For this POC, all loaded content is
 * passed directly to the LLM prompt rather than using vector similarity retrieval.
 *
 * Production note: replace in-memory list with a proper vector store (e.g.
 * SimpleVectorStore + Amazon Titan Embed) once the POC is validated, to support
 * semantic retrieval across a larger policy corpus.
 */
@Slf4j
@Service
public class DocumentIngestionService {

    private static final String DOCS_PATTERN = "classpath:docs/*.md";
    private static final int    MAX_CHUNK_CHARS = 1500;

    private final List<String> policyChunks = new ArrayList<>();

    /**
     * Triggered once the Spring application context is fully started.
     * Reads all .md files from classpath:docs/, splits them into chunks,
     * and stores them for prompt injection.
     */
    @EventListener(ApplicationReadyEvent.class)
    public void ingestDocuments() {
        log.info("DocumentIngestionService: scanning {} for policy documents...", DOCS_PATTERN);

        try {
            PathMatchingResourcePatternResolver resolver = new PathMatchingResourcePatternResolver();
            Resource[] resources = resolver.getResources(DOCS_PATTERN);

            if (resources.length == 0) {
                log.warn("DocumentIngestionService: no .md files found in classpath:docs/ — RAG context will be empty");
                return;
            }

            for (Resource resource : resources) {
                String filename = resource.getFilename();
                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(resource.getInputStream(), StandardCharsets.UTF_8))) {

                    String fullText = reader.lines().collect(Collectors.joining("\n"));
                    List<String> chunks = splitIntoChunks(fullText, MAX_CHUNK_CHARS);
                    policyChunks.addAll(chunks);
                    log.info("DocumentIngestionService: loaded '{}' -> {} chunk(s)", filename, chunks.size());

                } catch (Exception e) {
                    log.error("DocumentIngestionService: failed to read '{}': {}", filename, e.getMessage(), e);
                }
            }

            log.info("DocumentIngestionService: ingestion complete — {} total chunk(s) loaded from {} file(s)",
                    policyChunks.size(), resources.length);

        } catch (Exception e) {
            log.error("DocumentIngestionService: failed to scan docs directory: {}", e.getMessage(), e);
        }
    }

    /**
     * Returns all loaded policy chunks as an unmodifiable list.
     * Used by {@link BedrockExplainerService} to build the RAG context.
     *
     * @return unmodifiable list of policy text chunks; empty if ingestion failed
     */
    public List<String> getPolicyChunks() {
        return Collections.unmodifiableList(policyChunks);
    }

    /**
     * Returns all policy chunks joined into a single string, suitable for
     * direct injection into an LLM prompt.
     *
     * @return concatenated policy context string; empty string if no docs loaded
     */
    public String getPolicyContext() {
        if (policyChunks.isEmpty()) {
            return "No policy documents available.";
        }
        return String.join("\n\n---\n\n", policyChunks);
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /**
     * Splits a text string into chunks of at most {@code maxChars} characters,
     * breaking at paragraph boundaries (double newlines) where possible.
     */
    private List<String> splitIntoChunks(String text, int maxChars) {
        List<String> chunks = new ArrayList<>();
        String[] paragraphs = text.split("\n\n+");

        StringBuilder current = new StringBuilder();
        for (String paragraph : paragraphs) {
            if (current.length() + paragraph.length() + 2 > maxChars && current.length() > 0) {
                chunks.add(current.toString().trim());
                current = new StringBuilder();
            }
            if (paragraph.length() > maxChars) {
                // Paragraph itself exceeds limit — add as-is
                if (current.length() > 0) {
                    chunks.add(current.toString().trim());
                    current = new StringBuilder();
                }
                chunks.add(paragraph.trim());
            } else {
                if (current.length() > 0) current.append("\n\n");
                current.append(paragraph);
            }
        }
        if (current.length() > 0) {
            chunks.add(current.toString().trim());
        }
        return chunks.isEmpty() ? List.of(text.trim()) : chunks;
    }
}
