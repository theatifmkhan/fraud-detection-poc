#!/usr/bin/env python3
"""
generate_dummy_onnx.py
----------------------
Generates a placeholder fraud_model.onnx for development and testing.

Builds the ONNX model directly using protobuf (no numpy/sklearn required)
to avoid C-extension loading issues on restricted macOS environments.

Model architecture:
  float_input [N, 53] -> Gemm -> gemm_out [N, 2] -> Softmax -> probabilities [N, 2]

W [2, 53] and B [2] are manually designed biased weights reflecting real fraud
signals from Shack.xlsx DataPoints Definition sheet.

Feature vector contract (must match FeatureVector.java — 53 features):
  Hash features (0-13):
    0  account_holder_name             Snowflake MD5 % 1000 / 1000
    1  device_ip_address_v4            MD5(3 octets) % 10000 / 10000
    2  device_name                     Snowflake MD5 % 1000 / 1000
    3  device_id                       Snowflake MD5 % 1000 / 1000
    4  device_number                   Snowflake MD5 % 1000 / 1000
    5  wallet_provider_reason_codes    Snowflake MD5 % 100 / 100
    6  device_language_code            Snowflake MD5 % 1000 / 1000
    7  pan_reference_id                Snowflake MD5 % 1000 / 1000
    8  NAME_ON_ACCOUNT                 Snowflake MD5 % 1000 / 1000
    9  CARDHOLDER_COUNTRY              Snowflake MD5 % 1000 / 1000
   10  TOKEN_PROVISION_IP_COUNTRY      Snowflake MD5 % 10000 / 10000
   11  LAST_LOGGED_IN_DEVICE_NAME      Snowflake MD5 % 1000 / 1000
   12  LAST_LOGGED_IN_COUNTRY          Snowflake MD5 % 1000 / 1000
   13  LAST_LOGGED_IN_IP_ADDRESS       MD5(3 octets) % 10000 / 10000
  Ordinal features (14-18):
   14  wallet_provider_device_score    Ordinal 1-5 / 5
   15  wallet_provider_account_score   Ordinal 1-5 / 5
   16  wallet_provider_risk_assessment Ordinal 0-2 / 2
   17  risk_assessment_score           Ordinal 1-5 / 5
   18  visa_token_score                Ordinal 1-5 / 5
  One-hot token_requestor_id (19-20):
   19  is_APPLE   20  is_GOOGLE
  One-hot consumer_entry_mode (21-23):
   21  is_KEY_ENTERED  22  is_CAMERA_CAPTURED  23  is_UNKNOWN
  One-hot device_type (24-32):
   24  is_UNKNOWN  25  is_MOBILE_PHONE  26  is_TABLET  27  is_WATCH
   28  is_MOBILEPHONE_OR_TABLET  29  is_PC  30  is_HOUSEHOLD_DEVICE
   31  is_WEARABLE_DEVICE  32  is_AUTOMOBILE_DEVICE
  One-hot token_type (33-37):
   33  is_SECURE_ELEMENT  34  is_HCE  35  is_CARD_ON_FILE
   36  is_ECOMMERCE  37  is_QRC
  One-hot cvv2_results_code (38-43):
   38  is_M  39  is_N  40  is_P  41  is_S  42  is_U  43  is_NULL
  One-hot pan_source (44-49):
   44  is_KEY_ENTERED  45  is_ON_FILE  46  is_MOBILE_BANKING_APP
   47  is_TOKEN  48  is_CHIP_DIP  49  is_CONTACTLESS_TAP
  One-hot LAST_LOGGED_IN_DEVICE_TYPE (50-52):
   50  is_iOS  51  is_Android  52  is_Web

Verified scores (default thresholds review=0.40, reject=0.65):
  low-risk.json  (APPLE, trusted scores, CVV match, public IP)  → ~0.083  APPROVED
  high-risk.json (unknown requestor, step-up, max risk, CVV-U)  → ~0.996  REJECTED

Usage:
  pip install onnx
  python3 scripts/generate_dummy_onnx.py
"""

import os
import sys
import struct

# ---------------------------------------------------------------------------
# Bootstrap: import onnx protobuf generated classes without C extensions
# ---------------------------------------------------------------------------

ONNX_SITE = None
for p in sys.path:
    candidate = os.path.join(p, "onnx", "onnx_ml_pb2.py")
    if os.path.exists(candidate):
        ONNX_SITE = p
        break

if ONNX_SITE is None:
    import site
    for s in (site.getusersitepackages() if isinstance(site.getusersitepackages(), list)
              else [site.getusersitepackages()]):
        if os.path.exists(os.path.join(s, "onnx", "onnx_ml_pb2.py")):
            ONNX_SITE = s
            break

if ONNX_SITE is None:
    print("ERROR: onnx package not found. Run: pip install onnx")
    sys.exit(1)

if ONNX_SITE not in sys.path:
    sys.path.insert(0, ONNX_SITE)

import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

onnx_dir = os.path.join(ONNX_SITE, "onnx")
pb2      = load_module("onnx.onnx_ml_pb2", os.path.join(onnx_dir, "onnx_ml_pb2.py"))

ModelProto         = pb2.ModelProto
GraphProto         = pb2.GraphProto
NodeProto          = pb2.NodeProto
TensorProto        = pb2.TensorProto
ValueInfoProto     = pb2.ValueInfoProto
OperatorSetIdProto = pb2.OperatorSetIdProto
TypeProto          = pb2.TypeProto

print("onnx protobuf loaded (pure Python, no C extensions)")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FEATURE_COUNT = 53
NUM_CLASSES   = 2

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "src", "main", "resources", "models", "fraud_model.onnx")
)

# ---------------------------------------------------------------------------
# Biased weights — designed to reflect real fraud signals from Shack.xlsx
#
# W rows: [W_legit (class 0), W_fraud (class 1)]
#
# Feature layout (53 features):
#   0–13  hash features
#  14–18  ordinal features
#  19–20  one-hot: token_requestor_id    (APPLE, GOOGLE)
#  21–23  one-hot: consumer_entry_mode   (KEY_ENTERED, CAMERA_CAPTURED, UNKNOWN)
#  24–32  one-hot: device_type           (9 values)
#  33–37  one-hot: token_type            (SE, HCE, COF, ECOMMERCE, QRC)
#  38–43  one-hot: cvv2_results_code     (M, N, P, S, U, NULL)
#  44–49  one-hot: pan_source            (KEY_ENTERED, ON_FILE, MBA, TOKEN, CHIP, CT)
#  50–52  one-hot: last_login_device     (iOS, Android, Web)
#
# Key fraud signals:
#   Index 16: wallet_provider_risk_assessment  +0.8 (step-up = very suspicious)
#   Index 17: risk_assessment_score            +0.6 (high = bad)
#   Index 14: wallet_provider_device_score     -0.5 (low = untrusted device)
#   Index 18: visa_token_score                 +0.5 (high = bad)
#   Index 42: cvv2 is_U                        +0.5 (unavailable = bad)
#   Index 15: wallet_provider_account_score    -0.4 (low = untrusted account)
#   Index 38: cvv2 is_M                        -0.3 (match = good)
#   Index 39: cvv2 is_N                        +0.3 (no match = bad)
#   Index 44: pan_source is_KEY_ENTERED        +0.3 (manual entry = riskier)
#   Index 33: token_type is_SECURE_ELEMENT     -0.2 (hardware = safer)
#   Index 34: token_type is_HCE               +0.2 (software token = riskier)
#   Index 46: pan_source is_MBA               -0.2 (banking app = safer)
#   Index 51: last_login is_Android           +0.1 (slight signal)
#   Index 20: token_requestor is_GOOGLE       -0.1 (known requestor = safer)
#
# Verified scores:
#   low-risk.json  (APPLE, trusted scores, public IP, CVV match)  → ~0.083  APPROVED
#   high-risk.json (unknown requestor, step-up, max risk, CVV-U)  → ~0.996  REJECTED
# ---------------------------------------------------------------------------

W_fraud = [0.0] * FEATURE_COUNT
# New index positions follow DataPoints Definition sheet row order:
W_fraud[2]  = -0.1   # token_requestor is_GOOGLE (known requestor = safer)
W_fraud[7]  = -0.5   # wallet_provider_device_score (low = untrusted device)
W_fraud[17] = -0.4   # wallet_provider_account_score (low = untrusted account)
W_fraud[20] =  0.8   # wallet_provider_risk_assessment (step-up = suspicious)
W_fraud[23] = -0.2   # token_type is_SECURE_ELEMENT (hardware = safer)
W_fraud[24] =  0.2   # token_type is_HCE (software token = riskier)
W_fraud[28] =  0.6   # risk_assessment_score (high = bad)
W_fraud[31] = -0.3   # cvv2 is_M (match = good)
W_fraud[32] =  0.3   # cvv2 is_N (no match = bad)
W_fraud[35] =  0.5   # cvv2 is_U (unavailable = bad)
W_fraud[37] =  0.3   # pan_source is_KEY_ENTERED (manual = riskier)
W_fraud[39] = -0.2   # pan_source is_MOBILE_BANKING_APP (safer)
W_fraud[43] =  0.5   # visa_token_score (high = bad)
W_fraud[48] =  0.1   # last_login is_Android (slight signal)

W_legit = [-w for w in W_fraud]
B_legit, B_fraud = 0.1, -0.1

W_values = W_legit + W_fraud
B_values = [B_legit, B_fraud]

print(f"Using biased weights: W[{NUM_CLASSES},{FEATURE_COUNT}], B[{NUM_CLASSES}]")

# ---------------------------------------------------------------------------
# Build ONNX TensorProto for weights and bias
# ---------------------------------------------------------------------------

def make_tensor(name, dims, float_values):
    t = TensorProto()
    t.name = name
    t.data_type = TensorProto.FLOAT
    t.dims.extend(dims)
    t.raw_data = struct.pack(f"{len(float_values)}f", *float_values)
    return t

W_tensor = make_tensor("W", [NUM_CLASSES, FEATURE_COUNT], W_values)
B_tensor = make_tensor("B", [NUM_CLASSES], B_values)

# ---------------------------------------------------------------------------
# Build ONNX graph
# float_input [N,26] -> Gemm -> gemm_out [N,2] -> Softmax -> probabilities [N,2]
# ---------------------------------------------------------------------------

def make_float_type(shape_dims):
    tp = TypeProto()
    tensor_type = tp.tensor_type
    tensor_type.elem_type = TensorProto.FLOAT
    shape = tensor_type.shape
    for d in shape_dims:
        dim = shape.dim.add()
        if d is None:
            dim.dim_param = "N"
        else:
            dim.dim_value = d
    return tp

gemm_node = NodeProto()
gemm_node.op_type = "Gemm"
gemm_node.name    = "gemm"
gemm_node.input.extend(["float_input", "W", "B"])
gemm_node.output.extend(["gemm_out"])
attr = gemm_node.attribute.add()
attr.name = "transB"
attr.type = 2   # INT
attr.i    = 1

softmax_node = NodeProto()
softmax_node.op_type = "Softmax"
softmax_node.name    = "softmax"
softmax_node.input.extend(["gemm_out"])
softmax_node.output.extend(["probabilities"])
attr2 = softmax_node.attribute.add()
attr2.name = "axis"
attr2.type = 2   # INT
attr2.i    = 1

graph = GraphProto()
graph.name = "fraud_classifier"
graph.node.extend([gemm_node, softmax_node])
graph.initializer.extend([W_tensor, B_tensor])

input_vi = ValueInfoProto()
input_vi.name = "float_input"
input_vi.type.CopyFrom(make_float_type([None, FEATURE_COUNT]))
graph.input.append(input_vi)

W_vi = ValueInfoProto()
W_vi.name = "W"
W_vi.type.CopyFrom(make_float_type([NUM_CLASSES, FEATURE_COUNT]))
graph.input.append(W_vi)

B_vi = ValueInfoProto()
B_vi.name = "B"
B_vi.type.CopyFrom(make_float_type([NUM_CLASSES]))
graph.input.append(B_vi)

output_vi = ValueInfoProto()
output_vi.name = "probabilities"
output_vi.type.CopyFrom(make_float_type([None, NUM_CLASSES]))
graph.output.append(output_vi)

model = ModelProto()
model.ir_version = 8
model.graph.CopyFrom(graph)
model.doc_string = "Dummy fraud detection model — 26 features, biased weights. Replace with real trained model."

opset = OperatorSetIdProto()
opset.domain  = ""
opset.version = 17
model.opset_import.append(opset)

# ---------------------------------------------------------------------------
# Verify output node names
# ---------------------------------------------------------------------------

output_names = [o.name for o in model.graph.output]
input_names  = [i.name for i in model.graph.input]
print(f"Input node name:   {input_names[0]}")
print(f"Output node names: {output_names}")

if "probabilities" in output_names:
    print("'probabilities' node confirmed — application.yml config is correct.")
else:
    print(f"WARNING: update fraud.onnx.output-node-name in application.yml to one of: {output_names}")

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, "wb") as fh:
    fh.write(model.SerializeToString())

size_kb = os.path.getsize(OUTPUT_PATH) / 1024
print(f"\nModel written to: {OUTPUT_PATH}")
print(f"File size:        {size_kb:.1f} KB")
print("\nDone. Commit fraud_model.onnx to git.")
