# Fraud Detection POC — AI-Powered Tokenization Risk Engine

A proof-of-concept application that evaluates incoming card tokenization provisioning requests for fraud risk using a dual-path AI pipeline:

- **Path A — ONNX ML Inference:** A compiled XGBoost/RandomForest classifier runs sub-millisecond fraud probability scoring on every request
- **Path B — Amazon Bedrock RAG:** For high-risk decisions, an LLM (via Amazon Bedrock Converse API) generates a natural-language Analyst Audit Report grounded in internal fraud policy documents

The request schema mirrors the real Visa `approveProvisioning` API used in production (`visa-tokenization` service). The 53-feature vector is derived from `Shack.xlsx — DataPoints Definition` (26 source columns, expanded via one-hot encoding for all nominal categorical fields), ensuring training-serving parity with the Snowflake export pipeline.

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
        │       Converts raw TokenizationRequest into float[53]
        │       following DataPoints Definition sheet row order.
        │       Hash fields  → Snowflake-compatible MD5 % buckets
        │       Ordinal fields→ raw int / declared max
        │       Nominal fields→ one-hot binary expansion
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

## Feature Vector — 53 Features

Source: `Shack.xlsx — DataPoints Definition` sheet (26 source columns).
Index order follows the sheet **row by row**. One-hot expanded columns for each
nominal field appear immediately after their source row position.
This order is the strict contract between `EncodingService.java`,
the Snowflake training export SQL, and the trained ONNX model.

See also: `Shack.xlsx — Model Training Columns` for the exact Snowflake SQL
expressions and training column names.

### Encoding strategy

| Type | Fields | Encoding |
|------|--------|----------|
| **High-Cardinality / Free Text** | account_holder_name, device_name, device_id, device_number, device_language_code, pan_reference_id, name_on_account, cardholder_country, token_provision_ip_country, last_logged_in_device_name, last_logged_in_country, wallet_provider_reason_codes | Snowflake MD5 `ABS(MD5_NUMBER_LOWER64(value)) % buckets / buckets` |
| **IP Address** | device_ip_address_v4, last_logged_in_ip_address | MD5 of first 3 octets, 10 000 buckets |
| **Ordinal** | wallet_provider_device_score, wallet_provider_account_score, wallet_provider_risk_assessment, risk_assessment_score, visa_token_score | `raw integer / declared max` |
| **One-hot (nominal)** | token_requestor_id, consumer_entry_mode, device_type, token_type, cvv2_results_code, pan_source, LAST_LOGGED_IN_DEVICE_TYPE | One binary float per category value (1.0 or 0.0) |

### Full index table

| Index | Training Column Name | Source Field | Encoding |
|-------|---------------------|-------------|----------|
| 0 | `account_holder_name` | `accountHolderName` | MD5 % 1000 / 1000 |
| 1 | `token_requestor_is_apple` | `tokenRequestorId` | One-hot |
| 2 | `token_requestor_is_google` | `tokenRequestorId` | One-hot |
| 3 | `consumer_entry_is_key_entered` | `consumerEntryMode` | One-hot |
| 4 | `consumer_entry_is_camera_captured` | `consumerEntryMode` | One-hot |
| 5 | `consumer_entry_is_unknown` | `consumerEntryMode` | One-hot |
| 6 | `device_ip_address_v4` | `deviceInfo.deviceIpAddressV4` | MD5(3 octets) % 10000 / 10000 |
| 7 | `wallet_provider_device_score` | `walletProviderDeviceScore` | Ordinal 1–5 / 5 |
| 8 | `device_type_is_unknown` | `deviceInfo.deviceType` | One-hot |
| 9 | `device_type_is_mobile_phone` | `deviceInfo.deviceType` | One-hot |
| 10 | `device_type_is_tablet` | `deviceInfo.deviceType` | One-hot |
| 11 | `device_type_is_watch` | `deviceInfo.deviceType` | One-hot |
| 12 | `device_type_is_mobilephone_or_tablet` | `deviceInfo.deviceType` | One-hot |
| 13 | `device_type_is_pc` | `deviceInfo.deviceType` | One-hot |
| 14 | `device_type_is_household_device` | `deviceInfo.deviceType` | One-hot |
| 15 | `device_type_is_wearable_device` | `deviceInfo.deviceType` | One-hot |
| 16 | `device_type_is_automobile_device` | `deviceInfo.deviceType` | One-hot |
| 17 | `wallet_provider_account_score` | `walletProviderAccountScore` | Ordinal 1–5 / 5 |
| 18 | `device_name` | `deviceInfo.deviceName` | MD5 % 1000 / 1000 |
| 19 | `device_id` | `deviceInfo.deviceId` | MD5 % 1000 / 1000 |
| 20 | `wallet_provider_risk_assessment` | `walletProviderRiskAssessment` | Ordinal 0–2 / 2 |
| 21 | `device_number` | `deviceInfo.deviceNumber` | MD5 % 1000 / 1000 |
| 22 | `wallet_provider_reason_codes` | `walletProviderReasonCodes` | MD5 % 100 / 100 (multi-value, deferred) |
| 23 | `token_type_is_secure_element` | `tokenInfo.tokenType` | One-hot |
| 24 | `token_type_is_hce` | `tokenInfo.tokenType` | One-hot |
| 25 | `token_type_is_card_on_file` | `tokenInfo.tokenType` | One-hot |
| 26 | `token_type_is_ecommerce` | `tokenInfo.tokenType` | One-hot |
| 27 | `token_type_is_qrc` | `tokenInfo.tokenType` | One-hot |
| 28 | `risk_assessment_score` | `riskAssessmentScore` | Ordinal 1–5 / 5 |
| 29 | `device_language_code` | `deviceInfo.deviceLanguageCode` | MD5 % 1000 / 1000 |
| 30 | `pan_reference_id` | `panReferenceId` | MD5 % 1000 / 1000 |
| 31 | `cvv2_is_m` | `cvv2ResultsCode` | One-hot |
| 32 | `cvv2_is_n` | `cvv2ResultsCode` | One-hot |
| 33 | `cvv2_is_p` | `cvv2ResultsCode` | One-hot |
| 34 | `cvv2_is_s` | `cvv2ResultsCode` | One-hot |
| 35 | `cvv2_is_u` | `cvv2ResultsCode` | One-hot |
| 36 | `cvv2_is_null` | `cvv2ResultsCode` | One-hot |
| 37 | `pan_source_is_key_entered` | `panSource` | One-hot |
| 38 | `pan_source_is_on_file` | `panSource` | One-hot |
| 39 | `pan_source_is_mobile_banking_app` | `panSource` | One-hot |
| 40 | `pan_source_is_token` | `panSource` | One-hot |
| 41 | `pan_source_is_chip_dip` | `panSource` | One-hot |
| 42 | `pan_source_is_contactless_tap` | `panSource` | One-hot |
| 43 | `visa_token_score` | `visaTokenScore` | Ordinal 1–5 / 5 |
| 44 | `name_on_account` | `nameOnAccount` | MD5 % 1000 / 1000 |
| 45 | `cardholder_country` | `cardholderCountry` | MD5 % 1000 / 1000 |
| 46 | `token_provision_ip_country` | `tokenProvisionIpCountry` | MD5 % 10000 / 10000 |
| 47 | `last_login_device_is_ios` | `lastLoggedInDeviceType` | One-hot |
| 48 | `last_login_device_is_android` | `lastLoggedInDeviceType` | One-hot |
| 49 | `last_login_device_is_web` | `lastLoggedInDeviceType` | One-hot |
| 50 | `last_logged_in_device_name` | `lastLoggedInDeviceName` | MD5 % 1000 / 1000 |
| 51 | `last_logged_in_country` | `lastLoggedInCountry` | MD5 % 1000 / 1000 |
| 52 | `last_logged_in_ip_address` | `lastLoggedInIpAddress` | MD5(3 octets) % 10000 / 10000 |

---

## Snowflake-Compatible Hashing

All high-cardinality and free-text features use the same hashing function as Snowflake's `MD5_NUMBER_LOWER64`, ensuring identical bucket assignments at training time (Snowflake SQL) and serving time (Java `EncodingService`).

**Snowflake SQL** (the `AS` alias must use the exact training column name):
```sql
ABS(MD5_NUMBER_LOWER64(account_holder_name)) % 1000 / 1000.0 AS account_holder_name
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
│   └── generate_dummy_onnx.py            # Generates placeholder fraud_model.onnx (53 features)
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
    │   │   ├── TokenizationRequest.java   # 53-feature request — mirrors Visa approveProvisioning API
    │   │   ├── TokenInfo.java             # Nested token metadata
    │   │   ├── DeviceInfo.java            # Nested device metadata
    │   │   └── EvaluationResponse.java    # score + decision + aiExplanation
    │   ├── service/
    │   │   ├── EncodingService.java       # Raw request → float[53] (hash / ordinal / one-hot)
    │   │   ├── ModelInferenceService.java # ONNX Runtime inference wrapper
    │   │   ├── DocumentIngestionService.java # Loads fraud-policy.md at startup
    │   │   └── BedrockExplainerService.java  # Bedrock Converse RAG explanation
    │   └── util/
    │       └── FeatureVector.java         # Canonical feature index constants (53 features)
    └── resources/
        ├── application.yml
        ├── docs/
        │   └── fraud-policy.md            # Internal fraud rules — RAG context
        ├── mappings/
        │   └── encoding-mappings.json     # Hash bucket sizes and ordinal max values
        ├── models/
        │   └── fraud_model.onnx           # Placeholder model (replace with trained)
        └── static/
            ├── index.html                 # Dashboard — Tailwind + vanilla JS
            └── samples/
                ├── low-risk.json          # Sample APPROVED payload
                └── high-risk.json         # Sample REJECTED payload
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
| Latency | End-to-end request time shown in the top-right of the score card (e.g. `38 ms` for ONNX-only, `2.34 s` for ONNX + Bedrock) |
| ML Score | Fraud probability from the ONNX model (0.000–1.000), colour-coded gauge |
| Final Decision | `APPROVED` (green) / `REVIEW` (yellow) / `REJECTED` (red) |
| Analyst Audit Report | Natural-language explanation from Amazon Bedrock — visible only on `REJECTED` decisions |

---

## API Reference

### `POST /api/v1/evaluate`

**Request body** (`application/json`) — fields mirror the Visa `approveProvisioning` API.
All fields are optional at the HTTP level; `EncodingService` applies `0.0f` defaults for any
missing values so the model always receives a valid `float[53]` vector.

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
{ "score": 0.0832, "decision": "APPROVED" }
```

**Response — REJECTED (with Bedrock explanation):**
```json
{
  "score": 0.9957,
  "decision": "REJECTED",
  "aiExplanation": "This request was flagged due to..."
}
```

> `aiExplanation` is omitted from the response when the decision is not `REJECTED`.

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
| `low-risk.json` | `~0.083` | APPROVED |
| `high-risk.json` | `~0.996` | REJECTED |

To regenerate the placeholder:
```bash
pip install onnx
python3 scripts/generate_dummy_onnx.py
```

### Swapping in the Real Trained Model

1. Export your SageMaker XGBoost model to ONNX:
   ```python
   import tarfile
   from xgboost import XGBClassifier
   from skl2onnx import convert_sklearn
   from skl2onnx.common.data_types import FloatTensorType

   # Extract SageMaker artefact
   with tarfile.open('model.tar.gz') as t:
       t.extractall('.')

   clf = XGBClassifier()
   clf.load_model('xgboost-model')

   initial_type = [("float_input", FloatTensorType([None, 53]))]
   onnx_model = convert_sklearn(clf, initial_types=initial_type, target_opset=17)

   with open('fraud_model.onnx', 'wb') as f:
       f.write(onnx_model.SerializeToString())
   ```

2. Verify node names before deploying:
   ```python
   import onnx
   model = onnx.load('fraud_model.onnx')
   print("Inputs: ", [i.name for i in model.graph.input])
   print("Outputs:", [o.name for o in model.graph.output])
   # Must be: float_input and probabilities
   ```

3. Ensure the input node is named `float_input` (shape `[N, 53]`) and the output node is named `probabilities` (shape `[N, 2]`, where column 1 = P(fraud))

4. Drop the file into `src/main/resources/models/fraud_model.onnx`

5. The **feature column order must match `FeatureVector.java` indices 0–52** exactly — this is the contract between the Snowflake export SQL and `EncodingService`. Refer to `Shack.xlsx — Model Training Columns` for the exact SQL expressions.

No configuration changes required — input/output node names already match.

---

## SageMaker Training Notes

**Target column:** `is_fraud` (binary: 0 = legitimate, 1 = fraud). This column is **not** a model input — it is the training label only.

**Recommended algorithm:** XGBoost binary classifier
```
objective         = "binary:logistic"
scale_pos_weight  = count(negative) / count(positive)  # handles class imbalance
```

**Evaluation metrics:** AUC-ROC (primary), Precision/Recall at threshold 0.65, F1 score. Do not use accuracy — fraud datasets are typically 0.1–2% positive class.

**Snowflake export requirements:**
- Each row = one historical tokenization request
- 53 feature columns in the exact order of `FeatureVector.java` indices 0–52
- Hash fields pre-encoded in Snowflake using `ABS(MD5_NUMBER_LOWER64(value)) % buckets / buckets.0`
- Ordinal fields normalised by their declared max value
- Nominal categorical fields one-hot expanded into binary (0/1) columns — one column per category value
- Column names must match training column names in `Shack.xlsx — Model Training Columns` sheet
- Followed by the `is_fraud` target column (last)

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
