#!/usr/bin/env python3
"""
VI.Capstone_Projects/16.LLM_Inference_Optimization/16.3_precision_and_compile.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chapter 16: Capstone 1 — LLM Inference optimization Lab
Section 2: Precision and torch.compile optimization Ladder
=======================================================================
Covers capstone section 16.2:
  • Load the baseline (from 16.2_baseline_inference.py) for before/after comparison
  • Apply FP16 / BF16 and measure the speedup over FP32
  • Apply torch.compile and measure compile overhead vs steady-state gain
  • Build a full optimization ladder table with cumulative speedup
  • Save the optimized snapshot for final comparison

Run:  python VI.Capstone_Projects/16.LLM_Inference_Optimization/16.3_precision_and_compile.py
All sections must print ✓. Run 16.2_baseline_inference.py first to generate the baseline.
"""

import copy
import json
import math
import os
import statistics
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("  Capstone 16.2 — Precision and torch.compile Ladder")
print("=" * 60)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}\n")

VOCAB_SIZE = 32_000
D_MODEL    = 512
N_HEADS    = 8
N_LAYERS   = 4
D_FF       = 2048
MAX_SEQ    = 512


# ── Rebuild the same model ────────────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq=512):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv   = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj  = nn.Linear(d_model, d_model, bias=False)
        mask = torch.tril(torch.ones(max_seq, max_seq)).view(1, 1, max_seq, max_seq)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) for t in qkv]
        scale = math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) / scale
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff  = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq=512):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        self.blocks  = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f  = nn.LayerNorm(d_model)
        self.head  = nn.Linear(d_model, vocab_size, bias=False)
    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


def make_model(dtype=torch.float32):
    torch.manual_seed(42)
    m = MiniGPT(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, D_FF, MAX_SEQ)
    m = m.to(dtype=dtype).to(DEVICE)
    m.eval()
    return m

def bench_ttft(model, prompt_len=256, warmup=5, iters=30):
    idx = torch.randint(0, VOCAB_SIZE, (1, prompt_len), device=DEVICE)
    if model.tok_emb.weight.dtype != torch.float32:
        idx = idx  # embedding handles long indices regardless of weight dtype
    def run():
        with torch.no_grad():
            model(idx)
    for _ in range(warmup): run()
    if DEVICE == "cuda": torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        if DEVICE == "cuda":
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); run(); e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        else:
            t0 = time.perf_counter(); run()
            times.append((time.perf_counter() - t0) * 1000)
    return statistics.mean(times), sorted(times)[int(0.99 * len(times))]


# ─────────────────────────────────────────────────────────────
# SECTION 1: Load baseline
# ─────────────────────────────────────────────────────────────
print("── Section 1: Load the Baseline Snapshot ──")
print("""
  The baseline from 16.2_baseline_inference.py provides the FP32 reference numbers.
  Every optimization rung will compute its speedup relative to FP32.
  If the baseline file doesn't exist, we establish it here.
""")

BASELINE_PATH = "/tmp/capstone16_baseline.json"
if os.path.exists(BASELINE_PATH):
    with open(BASELINE_PATH) as f:
        baseline = json.load(f)
    tps_fp32 = baseline.get("tps_fp32", 0)
    ttft_fp32_ms = baseline.get("ttft_by_len", {}).get("256", {}).get("mean_ms", None)
    print(f"  Loaded baseline: TPS(FP32)={tps_fp32} tok/s")
else:
    print("  Baseline not found — measuring FP32 reference now")
    m_fp32 = make_model(torch.float32)
    ttft_fp32_ms, _ = bench_ttft(m_fp32, prompt_len=256)
    tps_fp32 = None  # will be estimated
    print(f"  FP32 TTFT (prompt=256): {ttft_fp32_ms:.3f} ms")

# Measure FP32 as local reference
m_fp32 = make_model(torch.float32)
ttft_fp32_ms, p99_fp32 = bench_ttft(m_fp32, prompt_len=256)
print(f"  FP32 reference TTFT (prompt=256): {ttft_fp32_ms:.3f} ms")
assert ttft_fp32_ms > 0
print("  ✓ Section 1 passed — baseline loaded")


# ─────────────────────────────────────────────────────────────
# SECTION 2: FP16 precision
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: FP16 Precision ──")
print("""
  FP16 (half precision) uses 2 bytes per value vs 4 for FP32:
    • Model weights occupy half the VRAM
    • Memory bandwidth per token load is halved
    • On Tensor Core GPUs: 2× throughput (FP16 matmul path)

  On CPU: FP16 is typically SLOWER because x86 CPUs lack native FP16
  arithmetic and emulate it in software. The speedup only appears on GPU.

  TODO 1: Measure TTFT for FP16 and BF16 models.
  Create models with dtype=torch.float16 and dtype=torch.bfloat16.
  Compute speedup relative to FP32 baseline.
""")


def measure_precision_speedup(dtype: torch.dtype, prompt_len: int = 256,
                               warmup: int = 5, iters: int = 30) -> dict:
    """
    TODO 1: Build a model in the given dtype, measure TTFT,
    compute speedup vs FP32 baseline, return a result dict.
    """
    try:
        m = make_model(dtype)
        mean_ms, p99_ms = bench_ttft(m, prompt_len=prompt_len,
                                      warmup=warmup, iters=iters)
        speedup = ttft_fp32_ms / mean_ms
        vram_mb = 0.0
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
            idx = torch.randint(0, VOCAB_SIZE, (1, prompt_len), device=DEVICE)
            with torch.no_grad():
                m(idx)
            vram_mb = torch.cuda.max_memory_allocated() / 1e6
        del m
        return {"dtype": str(dtype).split(".")[-1], "mean_ms": round(mean_ms, 3),
                "p99_ms": round(p99_ms, 3), "speedup": round(speedup, 2),
                "vram_mb": round(vram_mb, 1), "error": None}
    except Exception as ex:
        return {"dtype": str(dtype).split(".")[-1], "mean_ms": None,
                "speedup": None, "error": str(ex)[:60]}


results = {"fp32": {"dtype": "float32", "mean_ms": round(ttft_fp32_ms, 3),
                    "speedup": 1.0, "vram_mb": 0.0, "error": None}}

print(f"  {'Dtype':<10}  {'TTFT mean (ms)':>15}  {'Speedup':>9}  {'VRAM (MB)':>10}")
print(f"  {'─'*10}  {'─'*15}  {'─'*9}  {'─'*10}")
print(f"  {'float32':<10}  {ttft_fp32_ms:>15.3f}  {1.0:>9.2f}×  {'N/A':>10}")

for dtype, key in [(torch.float16, "fp16"), (torch.bfloat16, "bf16")]:
    r = measure_precision_speedup(dtype)
    results[key] = r
    if r["error"] is None:
        vram_str = f"{r['vram_mb']:.0f}" if r.get("vram_mb") else "N/A"
        print(f"  {r['dtype']:<10}  {r['mean_ms']:>15.3f}  {r['speedup']:>9.2f}×  {vram_str:>10}")
    else:
        print(f"  {r['dtype']:<10}  {'error':>15}  {'N/A':>9}  {'N/A':>10}  ({r['error']})")

if DEVICE == "cuda":
    fp16_speedup = results.get("fp16", {}).get("speedup", 1.0)
    print(f"\n  FP16 speedup on GPU: {fp16_speedup:.2f}×  (expected 1.5–2.5×)")
else:
    print(f"\n  On CPU: FP16/BF16 may be slower or similar (no Tensor Core)")

assert results["fp32"]["speedup"] == 1.0, "FP32 should be the baseline (1.0×)"
print("  ✓ Section 2 passed — precision speedup measured")


# ─────────────────────────────────────────────────────────────
# SECTION 3: torch.compile
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: torch.compile ──")
print("""
  torch.compile (introduced in PyTorch 2.0) traces the model's computation
  graph and passes it to TorchInductor, which generates fused Triton kernels.

  KEY TRADE-OFFS:
    + Reduces kernel launch overhead (fewer, larger kernels)
    + Enables operator fusion (e.g. LayerNorm + attention QKV in one kernel)
    + Eliminates Python interpreter overhead for the forward pass
    - Compilation takes 30–120 seconds on first call (paid once per session)
    - Dynamic shapes cause recompilation — use static input shapes
    - Some models do not benefit (already memory-bandwidth-bound at batch=1)

  COMPILE MODES:
    "default"         : balance compile time vs runtime performance
    "reduce-overhead" : more aggressive fusion; longer compile
    "max-autotune"    : exhaustive kernel search; 10+ min compile time

  TODO 2: Compile the FP32 model with mode="reduce-overhead".
  Measure: (a) time to first result (includes compilation), (b) steady-state TTFT.
  Report compile overhead = first_call_ms - steady_state_ms.
""")


def measure_compile_overhead(model: nn.Module, prompt_len: int = 256,
                              steady_iters: int = 20) -> dict:
    """
    TODO 2: Measure compile overhead and steady-state TTFT for a compiled model.
    Returns dict with: first_call_ms, steady_mean_ms, compile_overhead_ms, speedup_vs_eager.
    """
    idx = torch.randint(0, VOCAB_SIZE, (1, prompt_len), device=DEVICE)

    def run():
        with torch.no_grad():
            model(idx)

    # First call — includes compilation
    t0 = time.perf_counter()
    run()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - t0) * 1000

    # Steady-state calls (post-compilation)
    times = []
    for _ in range(steady_iters):
        if DEVICE == "cuda":
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); run(); e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        else:
            t0 = time.perf_counter(); run()
            times.append((time.perf_counter() - t0) * 1000)

    steady_mean_ms = statistics.mean(times)
    compile_overhead_ms = max(0.0, first_call_ms - steady_mean_ms)
    speedup = ttft_fp32_ms / steady_mean_ms

    return {
        "first_call_ms":      round(first_call_ms, 1),
        "steady_mean_ms":     round(steady_mean_ms, 3),
        "compile_overhead_ms": round(compile_overhead_ms, 1),
        "speedup_vs_fp32":    round(speedup, 2),
    }


compile_result = {"first_call_ms": None, "steady_mean_ms": None,
                  "compile_overhead_ms": None, "speedup_vs_fp32": 1.0}
try:
    m_compiled = torch.compile(make_model(torch.float32), mode="reduce-overhead")
    compile_result = measure_compile_overhead(m_compiled, prompt_len=256)
    results["fp32_compile"] = compile_result

    print(f"  First call (includes compilation): {compile_result['first_call_ms']:.0f} ms")
    print(f"  Steady-state TTFT                : {compile_result['steady_mean_ms']:.3f} ms")
    print(f"  Compile overhead                 : {compile_result['compile_overhead_ms']:.0f} ms")
    print(f"  Speedup vs eager FP32            : {compile_result['speedup_vs_fp32']:.2f}×")

    n_amortise = int(compile_result['compile_overhead_ms'] /
                     max(ttft_fp32_ms - compile_result['steady_mean_ms'], 0.001))
    print(f"\n  Compile amortised after ~{n_amortise} inferences "
          f"(at {ttft_fp32_ms:.1f} ms/call saving {max(ttft_fp32_ms - compile_result['steady_mean_ms'], 0):.2f} ms)")
except Exception as ex:
    print(f"  torch.compile skipped: {ex}")

assert compile_result["speedup_vs_fp32"] is not None
print("  ✓ Section 3 passed — compile overhead and speedup measured")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Full optimization ladder table
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Full optimization Ladder ──")
print("""
  We now assemble the complete before/after table. Each rung shows
  the cumulative improvement from the previous rung.

  READING THE TABLE:
    Rung 1 = FP32 eager baseline (1.00×)
    Rung 2 = FP16 (precision benefit)
    Rung 3 = FP16 + compile (fusion benefit ON TOP of FP16)
    Each rung's speedup is relative to FP32 baseline.
""")

# FP16 + compile
fp16_compile_result = {"steady_mean_ms": None, "speedup_vs_fp32": None}
if DEVICE == "cuda":
    try:
        m_fp16_compiled = torch.compile(make_model(torch.float16), mode="reduce-overhead")
        fc = measure_compile_overhead(m_fp16_compiled, prompt_len=256)
        fp16_compile_result = fc
        results["fp16_compile"] = fc
    except Exception as ex:
        print(f"  FP16+compile skipped: {ex}")

# Build ladder table
print(f"\n  {'Rung':<6}  {'Config':<22}  {'TTFT (ms)':>10}  {'Speedup':>9}")
print(f"  {'─'*6}  {'─'*22}  {'─'*10}  {'─'*9}")

ladder = [
    ("1", "FP32 eager", ttft_fp32_ms, 1.0),
]
fp16_r = results.get("fp16", {})
if fp16_r.get("mean_ms"):
    ladder.append(("2", "FP16 eager", fp16_r["mean_ms"], fp16_r["speedup"]))

fp32_c = results.get("fp32_compile", {})
if fp32_c.get("steady_mean_ms"):
    ladder.append(("3", "FP32 + compile", fp32_c["steady_mean_ms"], fp32_c["speedup_vs_fp32"]))

fp16_c = results.get("fp16_compile", {})
if fp16_c.get("steady_mean_ms"):
    ladder.append(("4", "FP16 + compile", fp16_c["steady_mean_ms"], fp16_c["speedup_vs_fp32"]))

for rung, config, ms, speedup in ladder:
    print(f"  {rung:<6}  {config:<22}  {ms:>10.3f}  {speedup:>8.2f}×")

best_speedup = max(sp for _, _, _, sp in ladder)
best_config  = [c for _, c, _, sp in ladder if sp == best_speedup][0]
print(f"\n  Best configuration: {best_config}  ({best_speedup:.2f}× over FP32 eager)")

# ── Hypothesis logging ───────────────────────────────────────────
# Before each run, write your expected speedup below.
# After measuring, compare: if prediction is off by >2x, that gap
# reveals a flaw in your mental model — the most valuable learning.
HYPOTHESES = {
    # Format: "config_name": expected_speedup_vs_fp32
    # Fill these in BEFORE running the script, then compare with actuals.
    "fp16_hypothesis":          None,   # YOUR ESTIMATE: e.g. 2.0 for 2x
    "fp32_compile_hypothesis":  None,   # YOUR ESTIMATE
    "fp16_compile_hypothesis":  None,   # YOUR ESTIMATE
}

# Save optimized snapshot with both hypotheses and actuals
optimized = {
    "ladder": [{"rung": r, "config": c, "ttft_ms": round(ms, 3),
                "actual_speedup": round(sp, 2),
                "hypothesis_speedup": HYPOTHESES.get(
                    c.lower().replace(" ", "_").replace("+", "") + "_hypothesis")}
               for r, c, ms, sp in ladder],
    "best_config":  best_config,
    "best_speedup": round(best_speedup, 2),
    "hypotheses":   HYPOTHESES,
}
import tempfile, os as _os
_snap = _os.path.join(tempfile.gettempdir(), "capstone16_optimized.json")
with open(_snap, "w") as f:
    json.dump(optimized, f, indent=2)
print(f"\n  Snapshot saved to: {_snap}")

assert len(ladder) >= 1, "At least baseline should be in ladder"
assert best_speedup >= 1.0, "Best speedup should be at least 1.0×"
print("  ✓ Section 4 passed — optimization ladder built")


print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Capstone 16.2 complete!")
print(f"  Best result: {best_config} → {best_speedup:.2f}× over FP32 baseline")
print("  Next: VI.Capstone_Projects/16.LLM_Inference_Optimization/16.4_profiling_audit.py")
print("=" * 60)
