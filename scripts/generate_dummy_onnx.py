#!/usr/bin/env python3
"""
generate_dummy_onnx.py
----------------------
Generates a minimal placeholder fraud_model.onnx for development and testing.

Builds the ONNX model directly using protobuf (no numpy/sklearn required)
to avoid C-extension loading issues on restricted macOS environments.

The model implements a simple linear classifier:
  probabilities = Softmax(Gemm(float_input, W, B))

Where W [2, 10] and B [2] are fixed random weights seeded for reproducibility.
Output shape: [N, 2] — column 0 = P(legitimate), column 1 = P(fraud).

This is a dummy model with fixed weights. Replace with the real trained
model once Snowflake export + SageMaker training is complete.

Usage:
  pip install onnx
  python3 scripts/generate_dummy_onnx.py

Feature vector index contract (must match FeatureVector.java):
  0  device_os_ios              one-hot
  1  device_os_android          one-hot
  2  hour_of_day                0-23
  3  day_of_week                1-7
  4  ip_first_octet             0-255
  5  ip_is_private              0 or 1
  6  num_active_tokens_norm     0.0-1.0
  7  num_suspended_tokens_norm  0.0-1.0
  8  client_wallet_hash_norm    0.0-1.0
  9  token_ref_id_len_norm      0.0-1.0
"""

import os
import sys
import struct
import random

# ---------------------------------------------------------------------------
# Bootstrap: import onnx protobuf generated classes without C extensions
# ---------------------------------------------------------------------------

ONNX_SITE = None
for p in sys.path:
    candidate = os.path.join(p, "onnx", "onnx_ml_pb2.py")
    if os.path.exists(candidate):
        ONNX_SITE = os.path.join(p)
        break

if ONNX_SITE is None:
    # Try common user install path
    import site
    for s in site.getusersitepackages() if isinstance(site.getusersitepackages(), list) else [site.getusersitepackages()]:
        candidate = os.path.join(s, "onnx", "onnx_ml_pb2.py")
        if os.path.exists(candidate):
            ONNX_SITE = s
            break

if ONNX_SITE is None:
    print("ERROR: onnx package not found. Run: pip install onnx")
    sys.exit(1)

if ONNX_SITE not in sys.path:
    sys.path.insert(0, ONNX_SITE)

# Import the pure-Python protobuf generated module directly (no C extension needed)
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

onnx_dir = os.path.join(ONNX_SITE, "onnx")
pb2      = load_module("onnx.onnx_ml_pb2",  os.path.join(onnx_dir, "onnx_ml_pb2.py"))

ModelProto      = pb2.ModelProto
GraphProto      = pb2.GraphProto
NodeProto       = pb2.NodeProto
TensorProto     = pb2.TensorProto
ValueInfoProto  = pb2.ValueInfoProto
OperatorSetIdProto = pb2.OperatorSetIdProto
TypeProto       = pb2.TypeProto

print("onnx protobuf loaded (pure Python, no C extensions)")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FEATURE_COUNT = 10
NUM_CLASSES   = 2
RANDOM_SEED   = 42

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "src", "main", "resources", "models", "fraud_model.onnx")
)

# ---------------------------------------------------------------------------
# Generate fixed random weights (seeded for reproducibility)
# ---------------------------------------------------------------------------

rng = random.Random(RANDOM_SEED)

def rand_floats(n):
    return [rng.gauss(0, 0.1) for _ in range(n)]

# W: [NUM_CLASSES, FEATURE_COUNT] = [2, 10]
W_values = rand_floats(NUM_CLASSES * FEATURE_COUNT)
# B: [NUM_CLASSES] = [2]
B_values = rand_floats(NUM_CLASSES)

print(f"Generated weights: W[{NUM_CLASSES},{FEATURE_COUNT}], B[{NUM_CLASSES}]")

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
#
# Graph: float_input [N,10] -> Gemm -> gemm_out [N,2] -> Softmax -> probabilities [N,2]
# ---------------------------------------------------------------------------

def make_float_type(shape_dims):
    tp = TypeProto()
    tensor_type = tp.tensor_type
    tensor_type.elem_type = TensorProto.FLOAT
    shape = tensor_type.shape
    for d in shape_dims:
        dim = shape.dim.add()
        if d is None:
            dim.dim_param = "N"   # dynamic batch dimension
        else:
            dim.dim_value = d
    return tp

# Nodes
gemm_node = NodeProto()
gemm_node.op_type = "Gemm"
gemm_node.name    = "gemm"
gemm_node.input.extend(["float_input", "W", "B"])
gemm_node.output.extend(["gemm_out"])
# transB=1 means W is used as W^T so shape is [2,10] x [10,N]^T -> [N,2]
# ONNX AttributeProto types: FLOAT=1, INT=2, STRING=3, TENSOR=4
attr = gemm_node.attribute.add()
attr.name  = "transB"
attr.type  = 2  # INT
attr.i     = 1

softmax_node = NodeProto()
softmax_node.op_type = "Softmax"
softmax_node.name    = "softmax"
softmax_node.input.extend(["gemm_out"])
softmax_node.output.extend(["probabilities"])
attr2 = softmax_node.attribute.add()
attr2.name = "axis"
attr2.type = 2  # INT
attr2.i    = 1

# Graph
graph = GraphProto()
graph.name = "fraud_classifier"
graph.node.extend([gemm_node, softmax_node])

# Initializers (weights)
graph.initializer.extend([W_tensor, B_tensor])

# Graph input
input_vi = ValueInfoProto()
input_vi.name = "float_input"
input_vi.type.CopyFrom(make_float_type([None, FEATURE_COUNT]))
graph.input.append(input_vi)

# Also declare W and B as graph inputs (required for older opset compatibility)
W_vi = ValueInfoProto()
W_vi.name = "W"
W_vi.type.CopyFrom(make_float_type([NUM_CLASSES, FEATURE_COUNT]))
graph.input.append(W_vi)

B_vi = ValueInfoProto()
B_vi.name = "B"
B_vi.type.CopyFrom(make_float_type([NUM_CLASSES]))
graph.input.append(B_vi)

# Graph output
output_vi = ValueInfoProto()
output_vi.name = "probabilities"
output_vi.type.CopyFrom(make_float_type([None, NUM_CLASSES]))
graph.output.append(output_vi)

# Model
model = ModelProto()
model.ir_version = 8
model.graph.CopyFrom(graph)
model.doc_string = "Dummy fraud detection model — replace with real trained model"

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

with open(OUTPUT_PATH, "wb") as f:
    f.write(model.SerializeToString())

size_kb = os.path.getsize(OUTPUT_PATH) / 1024
print(f"\nModel written to: {OUTPUT_PATH}")
print(f"File size:        {size_kb:.1f} KB")
print("\nDone. Commit fraud_model.onnx to git.")
