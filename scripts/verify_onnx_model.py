#!/usr/bin/env python3
"""
verify_onnx_model.py
--------------------
Verifies that fraud_model.onnx is compatible with the fraud-detection-poc app.

Checks:
  1. Input node name  = float_input
  2. Output node name = probabilities
  3. Input shape      = [N, 53]  (matches FeatureVector.FEATURE_COUNT)
  4. Output shape     = [N, 2]   (binary classifier — col 0=legit, col 1=fraud)
  5. Inference smoke test — dummy input produces valid probabilities

Usage:
  pip install onnx onnxruntime
  python3 scripts/verify_onnx_model.py
  python3 scripts/verify_onnx_model.py path/to/fraud_model.onnx
"""

import sys
import os

# ── Resolve model path ────────────────────────────────────────────────────────
# Default: src/main/resources/models/fraud_model.onnx (relative to this script)
# Override: pass a path as the first CLI argument
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'src', 'main', 'resources', 'models', 'fraud_model.onnx'
    )
)

EXPECTED_INPUT  = 'float_input'
EXPECTED_OUTPUT = 'probabilities'
FEATURE_COUNT   = 53   # must match FeatureVector.FEATURE_COUNT

# ── Dependency check ──────────────────────────────────────────────────────────
try:
    import onnx
    import onnxruntime as ort
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install onnx onnxruntime")
    sys.exit(1)

print(f"Verifying: {MODEL_PATH}\n")

if not os.path.exists(MODEL_PATH):
    print(f"FAIL: model file not found at {MODEL_PATH}")
    sys.exit(1)

# ── Load model ────────────────────────────────────────────────────────────────
model = onnx.load(MODEL_PATH)

inputs  = [i.name for i in model.graph.input]
outputs = [o.name for o in model.graph.output]

print(f"Input nodes:  {inputs}")
print(f"Output nodes: {outputs}")
print()

failures = []

# ── Check 1: input node name ──────────────────────────────────────────────────
if EXPECTED_INPUT in inputs:
    print(f"[PASS] Input node '{EXPECTED_INPUT}' found")
else:
    msg = f"[FAIL] Expected input node '{EXPECTED_INPUT}', got {inputs}"
    print(msg)
    failures.append(msg)

# ── Check 2: output node name ─────────────────────────────────────────────────
if EXPECTED_OUTPUT in outputs:
    print(f"[PASS] Output node '{EXPECTED_OUTPUT}' found")
else:
    msg = f"[FAIL] Expected output node '{EXPECTED_OUTPUT}', got {outputs}"
    print(msg)
    failures.append(msg)
    # Check for common alternative names and suggest the fix
    alternatives = [o for o in outputs if 'prob' in o.lower() or 'label' in o.lower()]
    if alternatives:
        print(f"       Hint: found '{alternatives}' — update fraud.onnx.output-node-name "
              f"in application.yml to match, or rename during ONNX export.")

# ── Check 3: input shape [N, 53] ──────────────────────────────────────────────
for inp in model.graph.input:
    if inp.name == EXPECTED_INPUT:
        try:
            dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            print(f"[INFO] Input shape: {dims}")
            if len(dims) >= 2 and dims[1] == FEATURE_COUNT:
                print(f"[PASS] Input shape [N, {FEATURE_COUNT}] — matches FeatureVector.FEATURE_COUNT")
            elif len(dims) >= 2:
                msg = (f"[FAIL] Input shape mismatch: expected [N, {FEATURE_COUNT}], "
                       f"got {dims}. Check that training data had {FEATURE_COUNT} features "
                       f"in the correct column order.")
                print(msg)
                failures.append(msg)
        except Exception as e:
            print(f"[WARN] Could not read input shape: {e}")

# ── Check 4 & 5: output shape + inference smoke test ─────────────────────────
try:
    session = ort.InferenceSession(MODEL_PATH)
    dummy   = [[0.0] * FEATURE_COUNT]
    result  = session.run([EXPECTED_OUTPUT], {EXPECTED_INPUT: dummy})
    probs   = result[0][0]   # first row: [P(legit), P(fraud)]

    print(f"[INFO] Output shape: {result[0].shape}")

    # Shape [N, 2]
    if result[0].shape[1] == 2:
        print("[PASS] Output shape [N, 2] — binary classifier confirmed")
    else:
        msg = f"[FAIL] Output shape {result[0].shape} — expected [N, 2] for binary classifier"
        print(msg)
        failures.append(msg)

    # Probabilities in [0, 1]
    if all(0.0 <= p <= 1.0 for p in probs):
        print("[PASS] Probabilities are in range [0.0, 1.0]")
    else:
        msg = f"[FAIL] Probabilities out of range: {probs}"
        print(msg)
        failures.append(msg)

    # Probabilities sum to 1
    prob_sum = sum(probs)
    if abs(prob_sum - 1.0) < 1e-4:
        print(f"[PASS] Probabilities sum to 1.0")
    else:
        msg = f"[FAIL] Probabilities do not sum to 1.0 (sum={prob_sum:.6f})"
        print(msg)
        failures.append(msg)

    print(f"\n[INFO] Dummy inference result:")
    print(f"       P(legitimate) = {probs[0]:.6f}  [index 0]")
    print(f"       P(fraud)      = {probs[1]:.6f}  [index 1]  ← used by ModelInferenceService")

except Exception as e:
    msg = f"[FAIL] Inference smoke test failed: {e}"
    print(msg)
    failures.append(msg)

# ── Final result ──────────────────────────────────────────────────────────────
print()
if not failures:
    print("=" * 60)
    print("ALL CHECKS PASSED — model is compatible with the app.")
    print("=" * 60)
    sys.exit(0)
else:
    print("=" * 60)
    print(f"FAILED — {len(failures)} check(s) did not pass:")
    for f in failures:
        print(f"  {f}")
    print("=" * 60)
    sys.exit(1)
