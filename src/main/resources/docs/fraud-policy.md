# Tokenization Fraud Detection — Security Policy & Fraud Definitions

## 1. Overview

This document defines the internal fraud detection policies and risk thresholds
applied to card tokenization provisioning requests. It is used as context by the
AI Fraud Analyst Copilot to explain why a specific request was flagged.

---

## 2. High-Risk Signal Definitions

### 2.1 Device OS Anomalies
A request where `deviceInfo.osType` is absent, null, or does not match a known
operating system (iOS, Android) is considered elevated risk. Unrecognised OS types
may indicate emulated environments or automated provisioning scripts.

### 2.2 Private IP Address
A provisioning request originating from an RFC-1918 private IP address
(10.x.x.x, 172.16–31.x.x, 192.168.x.x) is suspicious in production environments.
Legitimate tokenization requests should originate from public IP space. Private IPs
may indicate requests proxied through internal infrastructure to bypass controls.

### 2.3 Elevated Suspended Token Count
A `numberOfSuspendedTokensForPAN` value greater than 3 for a single PAN is a strong
fraud indicator. Suspension of existing tokens often precedes fraudulent re-provisioning
attempts to transfer card control to a new device.

### 2.4 High Active Token Concentration
A `numberOfActiveTokensForPAN` value greater than 5 indicates an unusually high
number of active tokens for a single PAN. Legitimate cardholders typically have
1–2 active tokens. High counts suggest enumeration or abuse.

### 2.5 After-Hours Requests
Provisioning requests received between 00:00–04:00 UTC exhibit statistically higher
fraud rates. Combined with other risk signals, late-night requests should be
escalated for manual review.

### 2.6 Weekend Provisioning
Requests submitted on Saturday (day 6) or Sunday (day 7) have a higher fraud rate
compared to weekday provisioning. This is particularly relevant when combined with
high suspended token counts.

### 2.7 Unknown Wallet Provider
A `tokenRequestorId` that does not correspond to a known, whitelisted wallet provider
(Apple Pay, Google Pay, Samsung Pay, Garmin Pay) should be treated as high risk.
Unrecognised requestors may indicate test environments being used against production
infrastructure.

### 2.8 PAN Source Mismatch
Requests with `panSource` set to `MANUAL_ENTRY` combined with a high suspended token
count are strongly associated with social engineering fraud, where a victim is tricked
into providing card details verbally.

---

## 3. Decision Thresholds

| Score Range   | Decision  | Action Required                                      |
|---------------|-----------|------------------------------------------------------|
| 0.00 – 0.39   | APPROVED  | No action. Request proceeds to provisioning.         |
| 0.40 – 0.64   | REVIEW    | Flag for async human review. Request may proceed.   |
| 0.65 – 1.00   | REJECTED  | Block provisioning. Generate AI explanation report. |

---

## 4. Escalation Rules

- Any REJECTED decision must be logged with the full request payload (excluding
  `encryptedData`) and the AI explanation in the audit trail.
- A score of -1.0 indicates an inference engine error. These cases must be
  escalated immediately to the on-call engineer and treated as REVIEW.
- Repeated REVIEW decisions from the same `clientWalletAccountId` within 24 hours
  should be automatically escalated to REJECTED on the third occurrence.

---

## 5. Known Fraud Patterns

### 5.1 Device Takeover (DTO)
Attacker provisions a new token on their device using stolen PAN data.
Signals: new deviceId, high suspended tokens for PAN, MANUAL_ENTRY pan source,
private IP, unrecognised OS type.

### 5.2 Account Enumeration
Automated scripts attempt to provision tokens for a large number of PANs.
Signals: high request velocity from single IP, sequential tokenReferenceIds,
after-hours timing, missing or spoofed deviceInfo fields.

### 5.3 Insider Threat via Test Infrastructure
Internal test credentials used against production tokenization endpoint.
Signals: private IP address, known test tokenRequestorId, missing deviceNumber.

---

## 6. Compliance Notes

- All fraud decisions and AI explanations are subject to PCI-DSS audit logging requirements.
- The AI explanation is advisory only and does not constitute a final compliance decision.
- Models must be retrained at minimum quarterly using updated Snowflake export data.
