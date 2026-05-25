#!/usr/bin/env python3
"""
6.PyTorch_Optimisation/6.2_quantization.py  ─  Chapter 6: Quantization
=======================================================================
Covers book section 6.2:
  • INT8 dynamic quantization with torch.quantization
  • Memory footprint comparison FP32 vs INT8
  • Accuracy impact measurement
  • Static quantization and INT4 concepts for LLM deployment

Run:  python II.GPU_Programming_and_Profiling/6.PyTorch_Optimisation/6.2_quantization.py
Note: quantization runs on CPU (torch.quantization.quantize_dynamic does
      not support CUDA directly; this is by design).
All sections must print ✓.
"""

import time
import torch
import torch.nn as nn

print("=" * 60)
print("  Exercise 6.2 — Quantization")
print("=" * 60)

# Note: dynamic quantization runs on CPU
print("  Note: this exercise runs on CPU (quantize_dynamic is CPU-only)\n")

# ─────────────────────────────────────────────────────────────
# SECTION 1: INT8 Dynamic Quantization
# ─────────────────────────────────────────────────────────────
print("── Section 1: INT8 Dynamic Quantization ──")
print("""
  Dynamic quantization converts the model weights to INT8 at inference
  time.  It requires no calibration data — quantisation parameters are
  computed on-the-fly from the activation statistics of each batch.

  This is the easiest form of quantisation to apply and works well for
  encoder/decoder models and RNNs.  It gives:
    - ~4x reduction in model parameter memory (FP32 → INT8)
    - 1.5–2x speedup on modern CPUs with INT8 instruction support
    - Minimal accuracy loss (typically <1% on GLUE tasks)

  torch.quantization.quantize_dynamic(model, layer_types, dtype)
    model       : the FP32 model to quantise
    layer_types : a set of layer classes to quantise, e.g. {nn.Linear}
    dtype       : quantisation dtype, torch.qint8 for INT8
""")

# Build a 4-layer MLP (same for fp32 and quantized)
def build_mlp():
    return nn.Sequential(
        nn.Linear(256, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 256), nn.ReLU(),
        nn.Linear(256, 64),
    )

fp32_model = build_mlp()
fp32_model.eval()

# TODO 1: Apply INT8 dynamic quantization.
#   Call torch.quantization.quantize_dynamic with:
#     model      = fp32_model
#     qconfig_spec = {nn.Linear}   (set of layer types to quantise)
#     dtype      = torch.qint8
#   Store the result in quantized_model.
quantized_model = None  # YOUR CODE HERE

assert quantized_model is not None, \
    "quantized_model must not be None. Did you call quantize_dynamic()?"

# Quick benchmark on CPU
x_bench = torch.randn(128, 256)
ITERS = 50

# FP32 timing
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(ITERS):
        fp32_model(x_bench)
fp32_ms = (time.perf_counter() - t0) / ITERS * 1000

# INT8 timing
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(ITERS):
        quantized_model(x_bench)
int8_ms = (time.perf_counter() - t0) / ITERS * 1000

speedup = fp32_ms / int8_ms
print(f"  FP32 inference  : {fp32_ms:.2f} ms")
print(f"  INT8 inference  : {int8_ms:.2f} ms")
print(f"  CPU speedup     : {speedup:.2f}x (may vary by CPU; ~1.5–2x expected)")
print("  ✓ Section 1 passed — INT8 dynamic quantization applied")

# ─────────────────────────────────────────────────────────────
# SECTION 2: Memory Footprint Comparison
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: Memory Footprint ──")
print("""
  INT8 quantization replaces 4-byte FP32 weights with 1-byte INT8 values.
  In theory this gives a 4x reduction in weight memory.  In practice,
  some overhead exists for quantisation parameters (scale and zero-point),
  so the actual reduction is closer to 3–4x.

  For LLM deployment, memory is often the binding constraint:
    A 7B-parameter FP16 model requires 14 GB of memory.
    INT8 quantization brings this to ~7 GB.
    INT4 brings it to ~3.5 GB — fitting on a single A100 40 GB.
""")

# TODO 2: Compute fp32_size_mb.
#   Sum (p.numel() * 4) for all parameters in fp32_model, then divide by 1e6.
fp32_size_mb = None  # YOUR CODE HERE → sum(p.numel()*4 for p in fp32_model.parameters()) / 1e6

assert fp32_size_mb is not None, "compute fp32_size_mb"
assert fp32_size_mb > 0, "fp32_size_mb must be positive"

# INT8 estimate (1 byte per element for weights; activations still float)
int8_size_mb = fp32_size_mb / 4

print(f"  FP32 model parameter memory : {fp32_size_mb:.2f} MB")
print(f"  INT8 estimate (~÷4)         : {int8_size_mb:.2f} MB")
print(f"  INT4 estimate (~÷8)         : {fp32_size_mb/8:.2f} MB")
print(f"  (This model is small; a 7B model would be ~14 GB → ~1.75 GB in INT4)")
print("  ✓ Section 2 passed — memory footprint comparison complete")

# ─────────────────────────────────────────────────────────────
# SECTION 3: Accuracy Impact
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: Accuracy Impact ──")
print("""
  Quantization introduces rounding error because the 256 representable
  INT8 values cannot exactly represent all FP32 values.  The maximum
  absolute error per element is approximately:
    max_error ≈ max_abs_weight / 128

  For a typical neural network, the output difference between FP32 and
  INT8 is small but non-zero.  We measure the mean absolute difference:
    diff = (fp32_out - int8_out.float()).abs().mean()

  A well-quantized model has diff < 1.0 for typical inputs.
  If diff is large, the model may need calibration (static quantization).
""")

x_eval = torch.randn(64, 256)

# TODO 3: Run both fp32_model and quantized_model on x_eval (using torch.no_grad()).
#   Compute diff = (fp32_out - q_out.float()).abs().mean().
#   Store in fp32_out, q_out, and diff.
fp32_out = None  # YOUR CODE HERE → fp32_model(x_eval)
q_out    = None  # YOUR CODE HERE → quantized_model(x_eval)
diff     = None  # YOUR CODE HERE → (fp32_out - q_out.float()).abs().mean()

assert fp32_out is not None, "fp32_out must not be None"
assert q_out    is not None, "q_out must not be None"
assert diff     is not None, "diff must not be None"
assert diff < 1.0, \
    f"Mean absolute error {diff:.4f} is suspiciously large. " \
    "Quantization should not drastically change outputs."

print(f"  FP32 output sample (first 5): {fp32_out[0, :5].detach().numpy()}")
print(f"  INT8 output sample (first 5): {q_out.float()[0, :5].detach().numpy()}")
print(f"  Mean absolute difference     : {diff.item():.6f}")
print(f"  (Expected < 1.0 for a well-quantized model)")
print("  ✓ Section 3 passed — accuracy impact measured")

# ─────────────────────────────────────────────────────────────
# SECTION 4: Quantization Concepts: Static and INT4
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Quantization Concepts: Static and INT4 ──")
print("""
  ── Static Quantization ──

  Dynamic quantization computes scale/zero-point at runtime from each
  batch, which adds overhead.  Static quantization pre-computes these
  from a calibration dataset:

    Step 1: Insert observer hooks into the model
            model_prepared = torch.quantization.prepare(model)
    Step 2: Run representative data through the model
            for xb in calibration_loader:
                model_prepared(xb)
    Step 3: Convert to static quantized model
            model_static = torch.quantization.convert(model_prepared)

  Result: weights AND activations are quantized at fixed points.
  Faster than dynamic quantization, but requires calibration data.
  If the calibration data is unrepresentative, accuracy degrades.

  ── INT4 Weight-Only Quantization ──

  INT4 quantization stores each weight in 4 bits (half of INT8).
  Used by: llama.cpp, bitsandbytes (GPTQ, AWQ), ExLlamaV2.

  For LLM deployment:
    FP16 7B model:  7B × 2 bytes = 14 GB
    INT8 7B model:  7B × 1 byte  = 7 GB
    INT4 7B model:  7B × 0.5 bytes = 3.5 GB

  With INT4, a 70B model fits in two A100s (2 × 40 GB) instead of ten.
  The accuracy trade-off: GPTQ INT4 on Llama-2 loses ~1-2 perplexity
  points vs FP16 — acceptable for most production use cases.

  Tools for INT4 quantization:
    bitsandbytes: pip install bitsandbytes
      from transformers import BitsAndBytesConfig
      config = BitsAndBytesConfig(load_in_4bit=True)
      model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=config)

    GPTQ (via AutoGPTQ):
      Post-training quantization using calibration data.
      Higher quality than bitsandbytes at INT4.

    AWQ (Activation-aware Weight Quantization):
      Preserves the most important (high-activation) weights at higher precision.
      State-of-the-art INT4 quality as of 2024.
""")

# Conceptual assertion: verify 7B model memory table is correct
model_7b_params = 7_000_000_000
fp16_gb  = model_7b_params * 2 / 1e9
int8_gb  = model_7b_params * 1 / 1e9
int4_gb  = model_7b_params * 0.5 / 1e9

print(f"  Memory for a 7B-parameter model:")
print(f"    FP16 : {fp16_gb:.1f} GB")
print(f"    INT8 : {int8_gb:.1f} GB  ({fp16_gb/int8_gb:.0f}x reduction vs FP16)")
print(f"    INT4 : {int4_gb:.1f} GB  ({fp16_gb/int4_gb:.0f}x reduction vs FP16)")

assert abs(int4_gb - 3.5) < 0.1, \
    f"INT4 7B model should be ~3.5 GB, got {int4_gb:.2f} GB"
print(f"  ✓ INT4 7B estimate verified: {int4_gb:.1f} GB")
print("  ✓ Section 4 passed — static and INT4 concepts reviewed")

print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 6.2 complete!")
print("  You now understand dynamic quantization, memory footprint,")
print("  accuracy impact, and INT4 for LLM deployment.")
print("  Next: II.GPU_Programming_and_Profiling/6.PyTorch_Optimisation/6.3_torch_compile.py")
print("=" * 60)
