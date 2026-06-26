# Fraud Detection POC — AI-Powered Tokenization Risk Engine

A proof-of-concept application that evaluates incoming card tokenization provisioning requests for fraud risk using a dual-path AI pipeline:

- **Path A — ONNX ML Inference:** A compiled XGBoost/RandomForest classifier runs sub-millisecond fraud probability scoring on every request
- **Path B — Amazon Bedrock RAG:** For high-risk decisions, an LLM (via Amazon Bedrock Converse API) generates a natural-language Analyst Audit Report grounded in internal fraud policy documents

The request schema mirrors the real Visa `approveProvisioning` API used in production (`visa-tokenization` service). The 26-feature vector is derived from `Shack.xlsx — DataPoints Definition`, ensuring training-serving parity with the Snowflake export pipeline.

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
        │       Converts raw TokenizationRequest into float[26]
        │       using Snowflake-compatible MD5 hashing
        │       (ABS(MD5_NUMBER_LOWER64(value)) % buckets)
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
                Loads fraud-policy.md via DocumentIngestionService
                Calls Amazon Bedrock Converse API (Amazon Nova / Claude)
                Returns natural-language Analyst Audit Report
```

---

## Feature Vector — 26 Features

Source: `Shack.xlsx — DataPoints Definition` sheet. The index order is the strict contract between `EncodingService.java`, the Snowflake export SQL, and the trained ONNX model.

| Index | Column | Source Field | Type | Encoding |
|-------|--------|-------------|------|----------|
| 0 | `account_holder_name` | `accountHolderName` | Free Text | Snowflake MD5 % 1000 / 1000 |
| 1 | `token_requestor_id` | `tokenRequestorId` | Low-Card | APPLE=0, GOOGLE=1, OTHER=2 / 2 |
| 2 | `consumer_entry_mode` | `consumerEntryMode` | Low-Card | KEY_ENTERED=0, CAMERA_CAPTURED=1, UNKNOWN=2 / 2 |
| 3 | `device_ip_address_v4` | `deviceInfo.deviceIpAddressV4` | High-Card IP | MD5(first 3 octets) % 10000 / 10000 |
| 4 | `wallet_provider_device_score` | `walletProviderDeviceScore` | Ordinal 1–5 | raw / 5 |
| 5 | `device_type` | `deviceInfo.deviceType` | Low-Card | UNKNOWN=0…AUTOMOBILE=8 / 8 |
| 6 | `wallet_provider_account_score` | `walletProviderAccountScore` | Ordinal 1–5 | raw / 5 |
| 7 | `device_name` | `deviceInfo.deviceName` | Free Text | Snowflake MD5 % 1000 / 1000 |
| 8 | `device_id` | `deviceInfo.deviceId` | High-Card | Snowflake MD5 % 1000 / 1000 |
| 9 | `wallet_provider_risk_assessment` | `walletProviderRiskAssessment` | Ordinal 0–2 | raw / 2 |
| 10 | `device_number` | `deviceInfo.deviceNumber` | High-Card | Snowflake MD5 % 1000 / 1000 |
| 11 | `wallet_provider_reason_codes` | `walletProviderReasonCodes` | Low-Card multi-value | Snowflake MD5 of full string % 100 / 100 |
| 12 | `token_type` | `tokenInfo.tokenType` | Low-Card | SECURE_ELEMENT=0, HCE=1, CARD_ON_FILE=2, ECOMMERCE=3, QRC=4 / 4 |
| 13 | `risk_assessment_score` | `riskAssessmentScore` | Ordinal 1–5 | raw / 5 |
| 14 | `device_language_code` | `deviceInfo.deviceLanguageCode` | High-Card | Snowflake MD5 % 1000 / 1000 |
| 15 | `pan_reference_id` | `panReferenceId` | High-Card | Snowflake MD5 % 1000 / 1000 |
| 16 | `cvv2_results_code` | `cvv2ResultsCode` | Low-Card | M=0, N=1, P=2, S=3, U=4, NULL=5 / 5 |
| 17 | `pan_source` | `panSource` | Low-Card | KEY_ENTERED=0, ON_FILE=1, MOBILE_BANKING_APP=2, TOKEN=3, CHIP_DIP=4, CONTACTLESS_TAP=5 / 5 |
| 18 | `visa_token_score` | `visaTokenScore` | Ordinal 1–5 | raw / 5 |
| 19 | `NAME_ON_ACCOUNT` | `nameOnAccount` | Free Text | Snowflake MD5 % 1000 / 1000 |
| 20 | `CARDHOLDER_COUNTRY` | `cardholderCountry` | High-Card | Snowflake MD5 % 1000 / 1000 |
| 21 | `TOKEN_PROVISION_IP_COUNTRY` | `tokenProvisionIpCountry` | High-Card | Snowflake MD5 % 10000 / 10000 |
| 22 | `LAST_LOGGED_IN_DEVICE_TYPE` | `lastLoggedInDeviceType` | Low-Card | iOS=0, Android=1, Web=2 / 2 |
| 23 | `LAST_LOGGED_IN_DEVICE_NAME` | `lastLoggedInDeviceName` | High-Card | Snowflake MD5 % 1000 / 1000 |
| 24 | `LAST_LOGGED_IN_COUNTRY` | `lastLoggedInCountry` | High-Card | Snowflake MD5 % 1000 / 1000 |
| 25 | `LAST_LOGGED_IN_IP_ADDRESS` | `lastLoggedInIpAddress` | High-Card IP | MD5(first 3 octets) % 10000 / 10000 |

---

## Snowflake-Compatible Hashing

All high-cardinality and free-text features use the same hashing function as Snowflake's `MD5_NUMBER_LOWER64`, ensuring identical bucket assignments at training time (Snowflake SQL) and serving time (Java `EncodingService`).

**Snowflake SQL:**
```sql
ABS(MD5_NUMBER_LOWER64(column_value)) % 1000 AS feature
```

**Java (EncodingService):**
```java
// 1. Compute MD5 digest
byte[] digest = MessageDigest.getInstance("MD5")
    .digest(value.getBytes(StandardCharsets.UTF_8));
// 2. Extract lower 64 bits (last 8 bytes of 16-byte digest)
long lower64 = ByteBuffer.wrap(digest, 8, 8).getLong();
// 3. Apply modulo
long bucket = Math.abs(lower64) % buckets;
```

**IP address fields** use the first 3 octets as the input string (e.g. `"192.168.1"` from `"192.168.1.55"`) with 10 000 buckets.

---

## Project Structure

```
fraud-detection-poc/
├── build.gradle                          # Gradle Groovy DSL — Spring Boot 3.5, Spring AI 1.1.8
├── settings.gradle
├── scripts/
│   └── generate_dummy_onnx.py            # Generates placeholder fraud_model.onnx
└── src/main/
    ├── java/com/solarisbank/frauddetection/
    │   ├── FraudDetectionApplication.java
    │   ├── config/
    │   │   ├── BedrockRegionConfig.java   # Sets aws.region system property at startup
    │   │   ├── CorsConfig.java            # CORS for /api/** (localhost only)
    │   │   └── OnnxConfig.java            # OrtEnvironment + OrtSession beans
    │   ├── controller/
    │   │   └── EvaluationController.java  # POST /api/v1/evaluate
    │   ├── dto/
    │   │   ├── TokenizationRequest.java   # 26-feature request — mirrors Visa approveProvisioning API
    │   │   ├── TokenInfo.java             # Nested token metadata
    │   │   ├── DeviceInfo.java            # Nested device metadata
    │   │   └── EvaluationResponse.java    # score + decision + aiExplanation
    │   ├── service/
    │   │   ├── EncodingService.java       # Raw request → float[26] with Snowflake-compatible MD5
    │   │   ├── ModelInferenceService.java # ONNX Runtime inference wrapper
    │   │   ├── DocumentIngestionService.java # Loads fraud-policy.md at startup
    │   │   └── BedrockExplainerService.java  # Bedrock Converse RAG explanation
    │   └── util/
    │       └── FeatureVector.java         # Canonical feature index constants (26 features)
    └── resources/
        ├── application.yml
        ├── docs/
        │   └── fraud-policy.md            # Internal fraud rules — RAG context
        ├── mappings/
        │   └── encoding-mappings.json     # Label maps, bucket sizes, ordinal max values
        ├── models/
        │   └── fraud_model.onnx           # Placeholder model (replace with trained)
        └── static/
            ├── index.html                 # Dashboard — Tailwind + vanilla JS
            └── samples/
                ├── low-risk.json          # Sample APPROVED payload (all 26 features)
                └── high-risk.json         # Sample REJECTED payload (all 26 features)
```

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Java | 21 | Homebrew: `brew install openjdk@21` |
| AWS credentials | STS temporary or IAM long-lived | Needed only for Bedrock AI explanation (REJECTED decisions) |
| Bedrock model access | `eu.amazon.nova-micro-v1:0` | Must be enabled in your AWS account (eu-central-1) |

No other local tools required — Gradle wrapper (`./gradlew`) is committed to the repo.

---

## Running Locally

### 1. Clone the repository

```bash
cd ~/Desktop/GithubRepos
git clone https://github.com/theatifmkhan/fraud-detection-poc.git
cd fraud-detection-poc
```

### 2. Set JAVA_HOME

```bash
export JAVA_HOME=/opt/homebrew/opt/openjdk@21
```

### 3. Export AWS credentials

The app resolves credentials via the AWS `DefaultCredentialsProvider` chain. The region is configured in `application.yml` and set as a system property at startup — no `AWS_DEFAULT_REGION` env var is needed.

```bash
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
export AWS_SESSION_TOKEN="your-session-token"   # required for STS temporary credentials
```

> **Note:** AWS credentials are only required for the Bedrock AI explanation, which is triggered solely for `REJECTED` decisions. Without credentials the app starts and scores requests normally via ONNX — only the `aiExplanation` field will contain an error message on REJECTED decisions.

### 4. Start the application

```bash
./gradlew bootRun
```

Wait for this in the output:

```
Started FraudDetectionApplication in X.X seconds
```

You will also see these confirming successful initialisation:

```
BedrockRegionConfig: aws.region system property set to 'eu-central-1'
OnnxConfig: OrtSession loaded from classpath:models/fraud_model.onnx
OnnxConfig: model inputs  -> [float_input]
OnnxConfig: model outputs -> [probabilities]
DocumentIngestionService: loaded 'fraud-policy.md' -> 3 chunk(s)
```

### 5. Open the dashboard

Navigate to **[http://localhost:8080](http://localhost:8080)** in your browser.

---

## Using the Dashboard

**Left panel — Input Form**

Fill in the tokenization request fields manually, or click one of the two sample payload buttons to pre-fill all fields instantly:

| Button | Scenario | Signal summary |
|--------|----------|---------------|
| **Low Risk Sample** | Apple Pay, iOS, public IP, device score 4, account score 4, risk assessment 0, CVV match | APPROVED |
| **High Risk Sample** | Unknown requestor, Android emulator, private IP, device score 1, account score 1, risk assessment 2 (step-up), CVV unavailable, max risk scores | REJECTED |

**Right panel — Results**

| Element | Description |
|---------|-------------|
| ML Score | Fraud probability from the ONNX model (0.000–1.000), colour-coded gauge |
| Final Decision | `APPROVED` (green) / `REVIEW` (yellow) / `REJECTED` (red) |
| Analyst Audit Report | Natural-language explanation from Amazon Bedrock — visible only on `REJECTED` decisions |

---

## API Reference

### `POST /api/v1/evaluate`

**Minimal request body:**

```json
{
  "tokenRequestorId": "APPLE",
  "tokenReferenceId": "DNITHE302836209",
  "panReferenceId": "FPAN-REF-00124",
  "clientWalletAccountId": "WALLET-ACC-88291",
  "walletAccountEmailAddressHash": "b94d27b9934d3e08",
  "encryptedData": "eyJhbGciOiJSU0EtT0FFUCJ9",
  "panSource": "MOBILE_BANKING_APP",
  "consumerEntryMode": "ON_FILE",
  "cvv2ResultsCode": "M",
  "deviceInfo": {
    "osType": "iOS",
    "deviceType": "MOBILE_PHONE",
    "deviceIpAddressV4": "85.214.132.10",
    "deviceId": "043E602BEF50800190291281539496303FF438",
    "deviceName": "iPhone 15 Pro",
    "deviceNumber": "4912345678901",
    "deviceLanguageCode": "eng"
  },
  "tokenInfo": {
    "tokenType": "SECURE_ELEMENT"
  },
  "walletProviderDeviceScore": 4,
  "walletProviderAccountScore": 4,
  "walletProviderRiskAssessment": 0,
  "walletProviderReasonCodes": "01",
  "riskAssessmentScore": 1,
  "visaTokenScore": 2,
  "accountHolderName": "John Smith",
  "nameOnAccount": "John Smith",
  "cardholderCountry": "DE",
  "tokenProvisionIpCountry": "DE",
  "lastLoggedInDeviceType": "iOS",
  "lastLoggedInDeviceName": "iPhone 15 Pro",
  "lastLoggedInCountry": "DE",
  "lastLoggedInIpAddress": "85.214.132.10"
}
```

**Response — APPROVED:**
```json
{ "score": 0.2850, "decision": "APPROVED" }
```

**Response — REJECTED (with Bedrock explanation):**
```json
{
  "score": 0.9881,
  "decision": "REJECTED",
  "aiExplanation": "This request was flagged due to..."
}
```

> `aiExplanation` is omitted from the response when the decision is not `REJECTED`.

All fields not present in the body default to `null` / `0` per `EncodingService` defaults — the model always receives a valid `float[26]` vector.

---

## Configuration Reference

All settings in `src/main/resources/application.yml`.

| Property | Default | Description |
|----------|---------|-------------|
| `fraud.threshold.review` | `0.40` | Score ≥ this → `REVIEW` |
| `fraud.threshold.reject` | `0.65` | Score ≥ this → `REJECTED` + Bedrock call |
| `fraud.onnx.model-path` | `classpath:models/fraud_model.onnx` | Path to ONNX model |
| `fraud.onnx.input-node-name` | `float_input` | ONNX graph input node |
| `fraud.onnx.output-node-name` | `probabilities` | ONNX graph output node |
| `spring.ai.bedrock.aws.region` | `eu-central-1` | AWS region for Bedrock |
| `spring.ai.bedrock.converse.chat.options.model` | `eu.amazon.nova-micro-v1:0` | Bedrock inference profile ID |

Override at runtime without changing the file:
```bash
./gradlew bootRun --args='--fraud.threshold.reject=0.50'
```

---

## About the ONNX Model

The committed `fraud_model.onnx` is a **placeholder** generated by `scripts/generate_dummy_onnx.py`. It is a linear classifier (`Gemm + Softmax`) with manually biased weights that reflect real fraud signals — it produces realistic scores for the two sample payloads but is not trained on real data.

**Verified scores:**

| Payload | Score | Decision |
|---------|-------|----------|
| `low-risk.json` | `~0.285` | APPROVED |
| `high-risk.json` | `~0.988` | REJECTED |

To regenerate the placeholder:
```bash
pip install onnx
python3 scripts/generate_dummy_onnx.py
```

### Swapping in the Real Trained Model

1. Export your SageMaker XGBoost model to ONNX:
   ```python
   from skl2onnx import convert_sklearn
   from skl2onnx.common.data_types import FloatTensorType

   initial_type = [("float_input", FloatTensorType([None, 26]))]
   onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=17)
   ```

2. Ensure the input node is named `float_input` (shape `[N, 26]`) and the output node is named `probabilities` (shape `[N, 2]`)

3. Drop the file into `src/main/resources/models/fraud_model.onnx`

4. The **feature column order must match `FeatureVector.java` indices 0–25** exactly — this is the contract between the Snowflake export SQL and `EncodingService`

No configuration changes required — input/output node names already match.

---

## SageMaker Training Notes

**Target column:** `is_fraud` (binary: 0 = legitimate, 1 = fraud)

**Recommended algorithm:** XGBoost binary classifier
```
objective = "binary:logistic"
scale_pos_weight = count(negative) / count(positive)  # handles class imbalance
```

**Evaluation metrics:** AUC-ROC (primary), Precision/Recall at threshold 0.65, F1 score. Do not use accuracy — fraud datasets are typically 0.1–2% positive class.

**Snowflake export:** each row must contain the 26 features encoded in the same order and using the identical Snowflake SQL as the `FeatureVector.java` contract (same MD5 lower-64-bit hash, same bucket sizes, same label integers).

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Port 8080 already in use` | `lsof -ti:8080 \| xargs kill -9` |
| `Unable to load region from any of the providers` | Update to latest code — `BedrockRegionConfig` sets the region automatically from `application.yml` |
| `Bedrock call failed: model identifier invalid` | Your AWS session token has expired — export fresh credentials |
| `Bedrock call failed: model not accessible` | Ensure `eu.amazon.nova-micro-v1:0` is enabled in your AWS account under eu-central-1 |
| `ONNX model file not found at startup` | Run `python3 scripts/generate_dummy_onnx.py` (requires `pip install onnx`) |
| Java toolchain error | Ensure `export JAVA_HOME=/opt/homebrew/opt/openjdk@21` before `./gradlew` |

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
