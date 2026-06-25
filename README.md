# Fraud Detection POC — AI-Powered Tokenization Risk Engine

A proof-of-concept application that evaluates incoming card tokenization provisioning requests for fraud risk using a dual-path AI pipeline:

- **Path A — ONNX ML Inference:** A compiled XGBoost/RandomForest classifier runs sub-millisecond fraud probability scoring on every request
- **Path B — Amazon Bedrock RAG:** For high-risk decisions, Claude (via Amazon Bedrock Converse API) generates a natural-language Analyst Audit Report grounded in internal fraud policy documents

The request schema mirrors the real Visa `approveProvisioning` API used in production (`visa-tokenization` service).

---

## Architecture

```
Browser (index.html)
        │
        │  POST /api/v1/evaluate
        ▼
EvaluationController
        │
        ├─ 1. EncodingService
        │       Converts raw TokenizationRequest into a
        │       float[10] feature vector (Snowflake-aligned encoding)
        │
        ├─ 2. ModelInferenceService
        │       Loads fraud_model.onnx at startup via ONNX Runtime
        │       Runs inference → fraud probability score [0.0–1.0]
        │
        ├─ 3. Decision Logic
        │       score < 0.40   → APPROVED
        │       0.40–0.64      → REVIEW
        │       score ≥ 0.65   → REJECTED
        │
        └─ 4. BedrockExplainerService  (REJECTED only)
                Loads fraud-policy.md context via DocumentIngestionService
                Calls Amazon Bedrock Converse API (Amazon Nova / Claude)
                Returns natural-language Analyst Audit Report
```

---

## Feature Vector

The model receives a fixed `float[10]` array. The canonical index order is defined in `FeatureVector.java` and must match the column order used during Snowflake data export and SageMaker training.

| Index | Feature | Source | Encoding |
|-------|---------|--------|----------|
| 0 | `device_os_ios` | `deviceInfo.osType` | 1.0 if iOS, else 0.0 |
| 1 | `device_os_android` | `deviceInfo.osType` | 1.0 if Android, else 0.0 |
| 2 | `hour_of_day` | Server ingestion time (UTC) | 0–23, default -1.0 |
| 3 | `day_of_week` | Server ingestion time (UTC) | 1–7 (ISO-8601), default -1.0 |
| 4 | `ip_first_octet` | `deviceInfo.deviceIpAddressV4` | Parsed int, default 0.0 |
| 5 | `ip_is_private` | `deviceInfo.deviceIpAddressV4` | 1.0 if RFC-1918, else 0.0 |
| 6 | `num_active_tokens_norm` | `tokenInfo.numberOfActiveTokensForPAN` | min(n, 20) / 20.0 |
| 7 | `num_suspended_tokens_norm` | `tokenInfo.numberOfSuspendedTokensForPAN` | min(n, 20) / 20.0 |
| 8 | `client_wallet_hash_norm` | `clientWalletAccountId` | abs(hash) % 1000 / 1000.0 |
| 9 | `token_ref_id_len_norm` | `tokenReferenceId` | min(len, 100) / 100.0 |

---

## Project Structure

```
fraud-detection-poc/
├── build.gradle                          # Gradle Groovy DSL — Spring Boot 3.5, Spring AI 1.1.8
├── settings.gradle
├── scripts/
│   └── generate_dummy_onnx.py            # Generates placeholder fraud_model.onnx (pure Python)
└── src/main/
    ├── java/com/solarisbank/frauddetection/
    │   ├── FraudDetectionApplication.java
    │   ├── config/
    │   │   ├── CorsConfig.java           # CORS for /api/** (localhost only)
    │   │   └── OnnxConfig.java           # OrtEnvironment + OrtSession beans
    │   ├── controller/
    │   │   └── EvaluationController.java # POST /api/v1/evaluate
    │   ├── dto/
    │   │   ├── TokenizationRequest.java  # Mirrors Visa approveProvisioning API
    │   │   ├── TokenInfo.java            # Nested token metadata
    │   │   ├── DeviceInfo.java           # Nested device metadata
    │   │   └── EvaluationResponse.java   # score + decision + aiExplanation
    │   ├── service/
    │   │   ├── EncodingService.java      # Raw request → float[10] feature vector
    │   │   ├── ModelInferenceService.java# ONNX Runtime inference wrapper
    │   │   ├── DocumentIngestionService.java # Loads fraud-policy.md at startup
    │   │   └── BedrockExplainerService.java  # Bedrock Converse RAG explanation
    │   └── util/
    │       └── FeatureVector.java        # Canonical feature index constants
    └── resources/
        ├── application.yml
        ├── docs/
        │   └── fraud-policy.md           # Internal fraud rules — RAG context
        ├── mappings/
        │   └── encoding-mappings.json    # Categorical encoding rules + defaults
        ├── models/
        │   └── fraud_model.onnx          # Placeholder model (replace with trained)
        └── static/
            ├── index.html                # Dashboard — Tailwind + vanilla JS
            └── samples/
                ├── low-risk.json         # Sample APPROVED payload
                └── high-risk.json        # Sample REJECTED payload
```

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Java | 21 | Homebrew: `brew install openjdk@21` |
| AWS credentials | STS temporary or IAM long-lived | Needs Bedrock access in `eu-central-1` |
| Bedrock model access | `eu.amazon.nova-micro-v1:0` | Must be enabled in your AWS account |

No other local tools required — Gradle wrapper (`./gradlew`) is committed to the repo.

---

## Running Locally

### 1. Clone the repository

```bash
cd ~/Desktop/GithubRepos    # or wherever you keep your repos
git clone https://github.com/theatifmkhan/fraud-detection-poc.git
cd fraud-detection-poc
```

### 2. Export AWS credentials

The application resolves credentials via the AWS `DefaultCredentialsProvider` chain. Export the three environment variables in your terminal session before starting the app:

```bash
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
export AWS_SESSION_TOKEN="your-session-token"   # required for STS temporary credentials
```

> **Note:** The Bedrock call (AI explanation) is only triggered for REJECTED decisions. If you only want to test the ONNX inference path you can run without valid credentials — the app will start and score requests, but the `aiExplanation` field will contain an error message.

### 3. Set JAVA_HOME to Java 21

```bash
export JAVA_HOME=/opt/homebrew/opt/openjdk@21
```

### 4. Start the application

```bash
./gradlew bootRun
```

Wait for this line in the output:

```
Started FraudDetectionApplication in X.X seconds
```

You will also see these confirming successful initialisation:

```
OnnxConfig: OrtSession loaded from classpath:models/fraud_model.onnx
OnnxConfig: model inputs  -> [float_input]
OnnxConfig: model outputs -> [probabilities]
DocumentIngestionService: loaded 'fraud-policy.md' -> 3 chunk(s)
```

### 5. Open the dashboard

Navigate to **[http://localhost:8080](http://localhost:8080)** in your browser.

---

## Using the Dashboard

The dashboard has two panels:

**Left panel — Input Form**

Fill in the tokenization request fields manually, or use one of the two sample payload buttons at the top:

| Button | Scenario | Expected result |
|--------|----------|----------------|
| **Low Risk Sample** | iOS device, public IP, 1 active token | `APPROVED`, no AI explanation |
| **High Risk Sample** | Android emulator, private IP, 18 active / 12 suspended tokens, unknown requestor | `REVIEW` at default thresholds |

> To force the High Risk sample to `REJECTED` (and trigger the Bedrock explanation), restart the app with a lower reject threshold:
> ```bash
> ./gradlew bootRun --args='--fraud.threshold.reject=0.50'
> ```

**Right panel — Results**

| Element | Description |
|---------|-------------|
| ML Score | Fraud probability from the ONNX model (0.000–1.000), shown as a colour-coded gauge |
| Final Decision | `APPROVED` (green) / `REVIEW` (yellow) / `REJECTED` (red) |
| Analyst Audit Report | Natural-language explanation from Amazon Bedrock — only visible on `REJECTED` decisions |

---

## API Reference

### `POST /api/v1/evaluate`

**Request body** (`application/json`):

```json
{
  "tokenRequestorId": "APPLE_PAY_REQUESTOR",
  "tokenReferenceId": "DNITHE302836209",
  "panReferenceId": "FPAN-REF-00124",
  "clientWalletAccountId": "WALLET-ACC-88291",
  "walletAccountEmailAddressHash": "b94d27b9934d3e08",
  "encryptedData": "eyJhbGciOiJSU0EtT0FFUCJ9",
  "panSource": "MOBILE_BANKING_APP",
  "consumerEntryMode": "ON_FILE",
  "locale": "en-DE",
  "lifeCycleTraceId": "LCT-20250625-001",
  "deviceInfo": {
    "osType": "iOS",
    "deviceType": "MOBILE_PHONE",
    "deviceIpAddressV4": "85.214.132.10",
    "deviceId": "043E602BEF50800190291281539496303FF438",
    "deviceName": "iPhone 15 Pro",
    "deviceNumber": "4912345678901"
  },
  "tokenInfo": {
    "numberOfActiveTokensForPAN": 1,
    "numberOfInactiveTokensForPAN": 0,
    "numberOfSuspendedTokensForPAN": 0,
    "tokenType": "SECURE_ELEMENT",
    "tokenAssuranceMethod": "OTP"
  }
}
```

**Response body**:

```json
{
  "score": 0.0094,
  "decision": "APPROVED"
}
```

```json
{
  "score": 0.7821,
  "decision": "REJECTED",
  "aiExplanation": "This request was flagged due to..."
}
```

> `aiExplanation` is omitted from the response entirely when the decision is not `REJECTED`.

---

## Configuration Reference

All settings are in `src/main/resources/application.yml`.

| Property | Default | Description |
|----------|---------|-------------|
| `fraud.threshold.review` | `0.40` | Score at or above this → `REVIEW` |
| `fraud.threshold.reject` | `0.65` | Score at or above this → `REJECTED` + Bedrock call |
| `fraud.onnx.model-path` | `classpath:models/fraud_model.onnx` | Path to the compiled ONNX model |
| `fraud.onnx.input-node-name` | `float_input` | ONNX graph input node name |
| `fraud.onnx.output-node-name` | `probabilities` | ONNX graph output node name |
| `spring.ai.bedrock.aws.region` | `eu-central-1` | AWS region for Bedrock |
| `spring.ai.bedrock.converse.chat.options.model` | `eu.amazon.nova-micro-v1:0` | Bedrock inference profile ID |

Any property can be overridden at startup without changing the file:

```bash
./gradlew bootRun --args='--fraud.threshold.reject=0.50'
```

---

## About the ONNX Model

The committed `fraud_model.onnx` is a **placeholder** generated by `scripts/generate_dummy_onnx.py`. It is a small linear classifier (`Gemm + Softmax`) with random weights and produces arbitrary scores — it exists only so the full inference pipeline can be exercised end-to-end before a real model is available.

To regenerate the placeholder:

```bash
pip install onnx
python3 scripts/generate_dummy_onnx.py
```

> **To use a real model:** export your trained SageMaker model as ONNX, ensure the input node is named `float_input` (shape `[N, 10]`) and the output node is named `probabilities` (shape `[N, 2]`), and drop the file into `src/main/resources/models/fraud_model.onnx`. The feature column order must match the `FeatureVector.java` contract exactly.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Port 8080 already in use` | `lsof -ti:8080 \| xargs kill -9` |
| `Bedrock call failed: model identifier invalid` | Your AWS session token has expired — export fresh credentials |
| `Bedrock call failed: model not accessible` | Ensure the Bedrock model `eu.amazon.nova-micro-v1:0` is enabled in your AWS account under eu-central-1 |
| `ONNX model file not found at startup` | Run `python3 scripts/generate_dummy_onnx.py` (requires `pip install onnx`) |
| `Java toolchain error` | Ensure `export JAVA_HOME=/opt/homebrew/opt/openjdk@21` before running `./gradlew` |
| Score always in REVIEW, never REJECTED | The placeholder model produces scores in the 0.40–0.62 range. Use `--args='--fraud.threshold.reject=0.50'` to trigger the REJECTED path for testing |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Runtime | Java 21, Spring Boot 3.5.0 |
| Build | Gradle 8.14.5 (Groovy DSL) |
| ML Inference | Microsoft ONNX Runtime 1.18.0 |
| LLM / RAG | Spring AI 1.1.8 + Amazon Bedrock Converse API |
| LLM Model | Amazon Nova Micro (`eu.amazon.nova-micro-v1:0`) via eu-central-1 cross-region inference profile |
| Frontend | Vanilla HTML + Tailwind CSS (CDN) + marked.js |
| AWS SDK | Embedded via Spring AI Bedrock starter (AWS SDK v2) |
