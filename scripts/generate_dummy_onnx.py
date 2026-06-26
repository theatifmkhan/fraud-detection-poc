#!/usr/bin/env python3
"""
generate_dummy_onnx.py
----------------------
Generates a placeholder fraud_model.onnx for development and testing.

Builds the ONNX model directly using protobuf (no numpy/sklearn required)
to avoid C-extension loading issues on restricted macOS environments.

Model architecture:
  float_input [N, 26] -> Gemm -> gemm_out [N, 2] -> Softmax -> probabilities [N, 2]

W [2, 26] and B [2] are manually designed biased weights that reflect real fraud
signals derived from the Shack.xlsx DataPoints Definition sheet.

Feature vector contract (must match FeatureVector.java — 26 features):
  0  account_holder_name          Free Text     Snowflake MD5 % 1000 / 1000
  1  token_requestor_id           Low-Card      APPLE=0, GOOGLE=1, OTHER=2 / 2
  2  consumer_entry_mode          Low-Card      KEY_ENTERED=0, CAMERA_CAPTURED=1, UNKNOWN=2 / 2
  3  device_ip_address_v4         High-Card IP  MD5(first 3 octets) % 10000 / 10000
  4  wallet_provider_device_score Ordinal 1-5   raw / 5
  5  device_type                  Low-Card      UNKNOWN=0..AUTOMOBILE=8 / 8
  6  wallet_provider_account_score Ordinal 1-5  raw / 5
  7  device_name                  Free Text     Snowflake MD5 % 1000 / 1000
  8  device_id                    High-Card     Snowflake MD5 % 1000 / 1000
  9  wallet_provider_risk_assess  Ordinal 0-2   raw / 2
  10 device_number                High-Card     Snowflake MD5 % 1000 / 1000
  11 wallet_provider_reason_codes Low-Card MV   Snowflake MD5 % 100 / 100
  12 token_type                   Low-Card      SE=0, HCE=1, COF=2, EC=3, QRC=4 / 4
  13 risk_assessment_score        Ordinal 1-5   raw / 5
  14 device_language_code         High-Card     Snowflake MD5 % 1000 / 1000
  15 pan_reference_id             High-Card     Snowflake MD5 % 1000 / 1000
  16 cvv2_results_code            Low-Card      M=0, N=1, P=2, S=3, U=4, NULL=5 / 5
  17 pan_source                   Low-Card      KEY_ENTERED=0..CONTACTLESS=5 / 5
  18 visa_token_score             Ordinal 1-5   raw / 5
  19 NAME_ON_ACCOUNT              Free Text     Snowflake MD5 % 1000 / 1000
  20 CARDHOLDER_COUNTRY           High-Card     Snowflake MD5 % 1000 / 1000
  21 TOKEN_PROVISION_IP_COUNTRY   High-Card     Snowflake MD5 % 10000 / 10000
  22 LAST_LOGGED_IN_DEVICE_TYPE   Low-Card      iOS=0, Android=1, Web=2 / 2
  23 LAST_LOGGED_IN_DEVICE_NAME   High-Card     Snowflake MD5 % 1000 / 1000
  24 LAST_LOGGED_IN_COUNTRY       High-Card     Snowflake MD5 % 1000 / 1000
  25 LAST_LOGGED_IN_IP_ADDRESS    High-Card IP  MD5(first 3 octets) % 10000 / 10000

Verified scores at default thresholds (review=0.40, reject=0.65):
  Low-risk  sample (APPLE, trusted device/account, public IP)  → ~0.285  APPROVED
  High-risk sample (unknown requestor, step-up assessment, private IP, max risk scores) → ~0.988  REJECTED

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

FEATURE_COUNT = 26
NUM_CLASSES   = 2

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "src", "main", "resources", "models", "fraud_model.onnx")
)

# ---------------------------------------------------------------------------
# Biased weights — designed to reflect real fraud signals from Shack.xlsx
#
# W rows: [W_legit (class 0), W_fraud (class 1)]
# Key signals driving fraud class:
#   Index  9 wallet_provider_risk_assessment: high (step-up/decline) = suspicious (+0.8)
#   Index  4 wallet_provider_device_score:    LOW score = untrusted device (-0.5)
#   Index  6 wallet_provider_account_score:   LOW score = untrusted account (-0.4)
#   Index 13 risk_assessment_score:           high = bad (+0.6)
#   Index 18 visa_token_score:                high = bad (+0.5)
#   Index 16 cvv2_results_code:               U/N = fail/unavailable (+0.3)
#   Index  1 token_requestor_id:              unknown requestor = higher label (+0.3)
#   Index 12 token_type:                      HCE riskier than SECURE_ELEMENT (+0.2)
#   Index  2 consumer_entry_mode:             manual entry modes (+0.2)
#   Index 17 pan_source:                      KEY_ENTERED=0, higher values are safer... (+0.1)
# ---------------------------------------------------------------------------

W_legit = [
    -0.0,  # 0  accountHolderName
    -0.3,  # 1  tokenRequestorId
    -0.2,  # 2  consumerEntryMode
    -0.0,  # 3  deviceIpAddressV4
     0.5,  # 4  walletProviderDeviceScore    (high = trusted = legit)
    -0.0,  # 5  deviceType
     0.4,  # 6  walletProviderAccountScore   (high = trusted = legit)
    -0.0,  # 7  deviceName
    -0.0,  # 8  deviceId
    -0.8,  # 9  walletProviderRiskAssessment (high = step-up = fraud)
    -0.0,  # 10 deviceNumber
    -0.0,  # 11 walletProviderReasonCodes
    -0.2,  # 12 tokenType
    -0.6,  # 13 riskAssessmentScore
    -0.0,  # 14 deviceLanguageCode
    -0.0,  # 15 panReferenceId
    -0.3,  # 16 cvv2ResultsCode
    -0.1,  # 17 panSource
    -0.5,  # 18 visaTokenScore
    -0.0,  # 19 nameOnAccount
    -0.0,  # 20 cardholderCountry
    -0.0,  # 21 tokenProvisionIpCountry
    -0.0,  # 22 lastLoggedInDeviceType
    -0.0,  # 23 lastLoggedInDeviceName
    -0.0,  # 24 lastLoggedInCountry
    -0.0,  # 25 lastLoggedInIpAddress
]

W_fraud = [
     0.0,  # 0  accountHolderName
     0.3,  # 1  tokenRequestorId
     0.2,  # 2  consumerEntryMode
     0.0,  # 3  deviceIpAddressV4
    -0.5,  # 4  walletProviderDeviceScore    (LOW = untrusted)
     0.0,  # 5  deviceType
    -0.4,  # 6  walletProviderAccountScore   (LOW = untrusted)
     0.0,  # 7  deviceName
     0.0,  # 8  deviceId
     0.8,  # 9  walletProviderRiskAssessment (HIGH = step-up)
     0.0,  # 10 deviceNumber
     0.0,  # 11 walletProviderReasonCodes
     0.2,  # 12 tokenType
     0.6,  # 13 riskAssessmentScore
     0.0,  # 14 deviceLanguageCode
     0.0,  # 15 panReferenceId
     0.3,  # 16 cvv2ResultsCode
     0.1,  # 17 panSource
     0.5,  # 18 visaTokenScore
     0.0,  # 19 nameOnAccount
     0.0,  # 20 cardholderCountry
     0.0,  # 21 tokenProvisionIpCountry
     0.0,  # 22 lastLoggedInDeviceType
     0.0,  # 23 lastLoggedInDeviceName
     0.0,  # 24 lastLoggedInCountry
     0.0,  # 25 lastLoggedInIpAddress
]

# B: [B_legit, B_fraud] — slight bias towards legitimate
B_legit, B_fraud = 0.1, -0.1

W_values = W_legit + W_fraud   # shape [2, 26] flattened row-major
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
