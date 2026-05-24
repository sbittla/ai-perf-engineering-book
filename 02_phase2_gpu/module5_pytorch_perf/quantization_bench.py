#!/usr/bin/env python3
"""
quantization_bench.py  ─  Phase 2 / Module 5: Quantization Benchmarking
=========================================================================

HOW TO RUN
    python quantization_bench.py                  # all experiments
    python quantization_bench.py --exp dynamic    # just INT8 dynamic
    python quantization_bench.py --exp compare    # side-by-side comparison


"""

import argparse, os, sys
import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("--exp", default="all",
    choices=["all","dynamic","static","compare","memory"])
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def cuda_ms(fn, warmup=5, iters=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

def sep(t): print(f"\n{'═'*60}\n  {t}\n{'─'*60}")

# ─────────────────────────────────────────────────────────────────────────────
# EXP 1 – Dynamic INT8 quantization
# ─────────────────────────────────────────────────────────────────────────────
def exp_dynamic():
    sep("EXP 1 – Dynamic INT8 Quantization")
    print("""
  torch.quantization.quantize_dynamic:
    - Converts Linear layers to INT8 at inference time
    - No calibration data required
    - Scales computed dynamically per-activation
    - CPU inference: often 2-4× faster than FP32
    - GPU: INT8 tensor cores on Turing+ (mostly useful on CPU)
    """)
    # Build a representative FFN block (similar to one transformer FFN layer)
    D = 2048
    ffn = nn.Sequential(
        nn.Linear(D, D * 4),
        nn.GELU(),
        nn.Linear(D * 4, D),
    )

    x_cpu = torch.randn(8, 64, D)   # (batch, seq, hidden)

    # FP32 baseline (CPU)
    ffn.eval()
    import time
    t0 = time.perf_counter()
    for _ in range(10):
        with torch.no_grad():
            _ = ffn(x_cpu)
    ms_fp32 = (time.perf_counter() - t0) / 10 * 1000

    # Dynamic INT8
    # qconfig_spec: which layer types to quantize
    # dtype: the weight dtype (torch.qint8)
    ffn_q = torch.quantization.quantize_dynamic(
        ffn,
        qconfig_spec={nn.Linear},
        dtype=torch.qint8,
    )
    ffn_q.eval()
    t0 = time.perf_counter()
    for _ in range(10):
        with torch.no_grad():
            out_q = ffn_q(x_cpu)
    ms_int8 = (time.perf_counter() - t0) / 10 * 1000

    # Accuracy check
    with torch.no_grad():
        out_fp32 = ffn(x_cpu)
    max_err = (out_fp32 - out_q).abs().max().item()

    print(f"  FP32 CPU  : {ms_fp32:.2f} ms")
    print(f"  INT8 CPU  : {ms_int8:.2f} ms  ({ms_fp32/ms_int8:.2f}× speedup)")
    print(f"  Max error : {max_err:.5f}  (< 0.01 is acceptable)")

    # Memory comparison
    def model_size_mb(m):
        return sum(p.numel() * p.element_size() for p in m.parameters()) / 1e6

    print(f"\n  FP32 model size : {model_size_mb(ffn):.1f} MB")
    print(f"  INT8 model size : ~{model_size_mb(ffn)/2:.1f} MB  (estimated)")

# ─────────────────────────────────────────────────────────────────────────────
# EXP 2 – Simulate INT4 / GPTQ memory benefit on GPU
# ─────────────────────────────────────────────────────────────────────────────
def exp_compare():
    sep("EXP 2 – Memory & Bandwidth: FP16 vs INT8 vs INT4 (GPU)")
    print("""
  For LLM INFERENCE the primary benefit of quantization is:
    1. Smaller model → fits in limited VRAM (e.g. RTX 4060 8GB)
    2. Less data fetched per token → higher throughput (memory-BW bound)
    3. More concurrent requests (larger KV cache fits)

  We simulate weight loading throughput for a 7B parameter model.
    """)
    if DEVICE != "cuda":
        print("  (Skipped – CUDA not available)"); return

    # Simulate loading weights at different precisions (scaled down to fit 8GB GPU)
    # In real LLM decode: GPU reads ~all weights once per token
    n_weights = 4 * 1024 * 1024   # 16MB of FP32 = 4M floats

    configs = [
        ("FP32",  torch.float32, 4, 1.0),
        ("FP16",  torch.float16, 2, 1.0),
        ("INT8",  torch.int8,    1, 0.99),   # ~1% accuracy loss
        # INT4: PyTorch doesn't have a native int4 dtype, so we simulate
        # by halving INT8 sizes (2 weights packed per byte)
    ]

    print(f"  {'Dtype':>6}  {'Size(GB)':>10}  {'Load time(ms)':>14}  "
          f"{'Eff BW(GB/s)':>13}  {'Rel size':>10}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*14}  {'─'*13}  {'─'*10}")

    for name, dt, bytes_per, _ in configs:
        n = n_weights
        if dt == torch.int8:
            n = n_weights * 4  # 1 byte vs 4 bytes: 4× more ints for same memory footprint
        # Use rand-based creation to avoid int64 intermediate (torch.randint default)
        if dt == torch.int8:
            t = (torch.rand(n, device=DEVICE) * 127).to(dt)
        else:
            t = torch.rand(n, device=DEVICE).to(dt)
        size_gb = t.numel() * t.element_size() / 1e9
        ms = cuda_ms(lambda: t.sum())    # force memory read
        bw = size_gb / (ms / 1000)
        rel = bytes_per / 4
        print(f"  {name:>6}  {size_gb:>10.2f}  {ms:>14.3f}  {bw:>13.1f}  {rel:>9.2f}×")
        del t

    # INT4 simulation (pack 2 weights per byte)
    n_int4 = n_weights * 4 // 2   # half the INT8 count
    t4 = torch.randint(0, 127, (n_int4,), device=DEVICE, dtype=torch.int8)
    size4 = t4.numel() * 1 / 1e9
    ms4 = cuda_ms(lambda: t4.sum())
    bw4 = size4 / (ms4 / 1000)
    print(f"  {'INT4':>6}  {size4:>10.2f}  {ms4:>14.3f}  {bw4:>13.1f}  {'0.50':>10}×  ← simulated")
    del t4

    print(f"""
  KEY INSIGHT FOR RTX 4060 (8GB):
    Llama-2-7B weights:
      FP16  : 14 GB  → DOESN'T FIT (8GB GPU)
      INT8  :  7 GB  → just fits (no KV cache room)
      INT4  :  3.5GB → fits + room for KV cache of ~10 concurrent requests

  To use quantized models:
    # HuggingFace bitsandbytes:
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        load_in_8bit=True,      # INT8 via bitsandbytes
        # or:
        load_in_4bit=True,      # NF4 via bitsandbytes
        bnb_4bit_compute_dtype=torch.float16,
    )

    # vLLM with GPTQ/AWQ:
    python -m vllm.entrypoints.openai.api_server \\
        --model TheBloke/Llama-2-7B-GPTQ \\
        --quantization gptq
    """)

# ─────────────────────────────────────────────────────────────────────────────
# EXP 3 – Post-training quantization accuracy impact
# ─────────────────────────────────────────────────────────────────────────────
def exp_memory():
    sep("EXP 3 – Quantization Accuracy Impact (Perplexity Proxy)")
    print("""
  We can't run full LLM perplexity evaluation here, but we can measure
  the output difference between FP32 and INT8 as a proxy.

  In practice:
    INT8 (dynamic)  → perplexity increase < 0.5 for most models
    INT8 (static)   → perplexity increase < 0.2 with good calibration
    GPTQ INT4       → perplexity increase ~0.5-1.0 depending on model
    AWQ  INT4       → perplexity increase ~0.2-0.5 (better than GPTQ)
    """)
    D = 512
    model_fp32 = nn.Sequential(
        nn.Embedding(1000, D),
        nn.Linear(D, D * 4), nn.GELU(), nn.Linear(D * 4, D),
        nn.Linear(D, 1000),
    )
    model_fp32.eval()

    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32, {nn.Linear}, dtype=torch.qint8
    )
    model_int8.eval()

    ids = torch.randint(0, 1000, (4, 32))
    with torch.no_grad():
        logits_fp32 = model_fp32(ids)
        logits_int8 = model_int8(ids)

    diff      = (logits_fp32 - logits_int8).abs()
    max_diff  = diff.max().item()
    mean_diff = diff.mean().item()
    cos_sim   = torch.nn.functional.cosine_similarity(
        logits_fp32.reshape(-1), logits_int8.reshape(-1), dim=0
    ).item()

    print(f"  Output max  |FP32 - INT8| : {max_diff:.6f}")
    print(f"  Output mean |FP32 - INT8| : {mean_diff:.6f}")
    print(f"  Cosine similarity          : {cos_sim:.6f}  (1.0 = identical)")
    print(f"\n  Cosine > 0.999 → output nearly identical → safe to quantize")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  quantization_bench.py  ─  Phase 2 Module 5")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}")

    d = {"dynamic": exp_dynamic, "compare": exp_compare, "memory": exp_memory}
    if args.exp == "all":
        for fn in d.values(): fn()
    else:
        d[args.exp]()
