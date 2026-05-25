#!/usr/bin/env python3
"""
VI.Capstone_Projects/16.LLM_Inference_Optimisation/16.1_baseline_inference.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chapter 16: Capstone 1 — LLM Inference Optimisation Lab
Section 1: Establishing the Baseline
=======================================================================
Covers capstone section 16.1:
  • Build a GPT-2-style mini transformer for benchmarking
  • Measure TTFT (Time To First Token) — latency for the prefill phase
  • Measure TPS (Tokens Per Second) — decode throughput with KV cache
  • Measure peak GPU VRAM footprint
  • Save a structured baseline JSON as the "before" snapshot

Run:  python VI.Capstone_Projects/16.LLM_Inference_Optimisation/16.1_baseline_inference.py
All sections must print ✓.
"""

import json
import math
import statistics
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("  Capstone 16.1 — LLM Inference Baseline")
print("=" * 60)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}\n")


# ─────────────────────────────────────────────────────────────
# SECTION 1: The Mini Transformer
# ─────────────────────────────────────────────────────────────
print("── Section 1: Mini GPT-2 Style Transformer ──")
print("""
  For this capstone we use a compact GPT-2-style transformer that
  is large enough to show meaningful timing differences but small
  enough to run anywhere — including CPU-only machines.

  ARCHITECTURE:
    Vocab size : 32,000 (matches LLaMA tokenizer)
    n_layers   : 4       (vs GPT-2's 12)
    n_heads    : 8
    d_model    : 512
    d_ff       : 2048    (4× d_model, standard FFN ratio)

  This is a 28M parameter model. A production GPT-2 (117M) or
  LLaMA-7B (7B) will show larger speedups from the same techniques,
  but the bottlenecks and measurement approach are identical.

  PHASES:
    Prefill : forward pass over the full prompt (input tokens)
    Decode  : autoregressive generation, one token per step

  Both phases are measured separately because they have different
  bottlenecks: prefill is compute-bound, decode is memory-BW-bound.
""")


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq: int = 512):
        super().__init__()
        assert d_model % n_heads == 0
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
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
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
    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, max_seq: int = 512):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        self.blocks  = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.ln_f  = nn.LayerNorm(d_model)
        self.head  = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


VOCAB_SIZE = 32_000
D_MODEL    = 512
N_HEADS    = 8
N_LAYERS   = 4
D_FF       = 2048
MAX_SEQ    = 512

torch.manual_seed(42)
model = MiniGPT(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, D_FF, MAX_SEQ).to(DEVICE)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
param_mb  = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
print(f"  Model: {n_params/1e6:.1f}M parameters  {param_mb:.0f} MB (FP32)")
assert n_params > 1e6, "Model should have at least 1M parameters"
print("  ✓ Section 1 passed — mini GPT built and on device")


# ─────────────────────────────────────────────────────────────
# SECTION 2: Measuring TTFT (Time To First Token)
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: TTFT — Time To First Token ──")
print("""
  TTFT is the latency from receiving the user's request to producing
  the first output token. It is dominated by the PREFILL phase:
  a single forward pass over all prompt tokens.

  For a prompt of length L: TTFT ≈ forward_pass_latency(batch=1, seq=L)

  TTFT scales with prompt length because:
    - The attention matrix is L × L → O(L²) compute
    - The FFN processes L tokens in parallel → O(L) compute
  The dominant term is O(L²) for long prompts.

  TODO 1: Implement measure_ttft(model, prompt_len, iters=20)
  that generates a random token sequence of length prompt_len and
  times the forward pass. Return (mean_ms, p99_ms).
""")


def measure_ttft(model: nn.Module, prompt_len: int,
                 warmup: int = 5, iters: int = 20) -> tuple:
    """
    TODO 1: Measure TTFT for a prompt of prompt_len tokens.
    Input: random integer tensor of shape (1, prompt_len).
    Time the forward pass with CUDA events (GPU) or perf_counter (CPU).
    Return (mean_ms, p99_ms).
    """
    idx = torch.randint(0, VOCAB_SIZE, (1, prompt_len), device=DEVICE)

    def run():
        with torch.no_grad():
            model(idx)

    for _ in range(warmup):
        run()
    if DEVICE == "cuda":
        torch.cuda.synchronize()

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


prompt_lengths = [64, 128, 256, 512]
ttft_results = {}

print(f"  {'Prompt len':>12}  {'TTFT mean (ms)':>15}  {'TTFT P99 (ms)':>14}")
print(f"  {'─'*12}  {'─'*15}  {'─'*14}")
for plen in prompt_lengths:
    mean_ms, p99_ms = measure_ttft(model, plen)
    ttft_results[plen] = {"mean_ms": round(mean_ms, 3), "p99_ms": round(p99_ms, 3)}
    print(f"  {plen:>12}  {mean_ms:>15.3f}  {p99_ms:>14.3f}")

# TTFT should grow with prompt length (more compute)
ttft_means = [ttft_results[p]["mean_ms"] for p in prompt_lengths]
assert ttft_results[512]["mean_ms"] > ttft_results[64]["mean_ms"], \
    "TTFT should increase with prompt length"
print("  ✓ Section 2 passed — TTFT measured across prompt lengths")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Measuring TPS (Tokens Per Second)
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: TPS — Tokens Per Second ──")
print("""
  TPS measures the decode throughput: how many tokens the model can
  generate per second in the autoregressive decode loop.

  Each decode step:
    1. Loads all model weights from HBM (memory-bandwidth-bound)
    2. Computes attention over all past tokens (grows with length)
    3. Samples one token from the logit distribution

  We simulate the decode loop by running N_DECODE forward passes
  with a growing context, then compute TPS = N_DECODE / total_time.

  TODO 2: Implement measure_tps(model, prompt_len, n_decode=50)
  that simulates N_DECODE autoregressive steps and returns TPS.
  At each step, append the predicted token to the context.
""")


def measure_tps(model: nn.Module, prompt_len: int = 64,
                n_decode: int = 50) -> float:
    """
    TODO 2: Simulate autoregressive decoding; return tokens per second.
    Start with a random prompt of prompt_len tokens.
    For each of n_decode steps:
      - Run forward pass, take argmax of last logit as next token
      - Append to context (context grows by 1 each step)
    TPS = n_decode / total_time_s
    Return TPS.
    """
    ctx = torch.randint(0, VOCAB_SIZE, (1, prompt_len), device=DEVICE)

    # Warmup: 3 steps
    for _ in range(3):
        with torch.no_grad():
            logits = model(ctx)
        next_tok = logits[:, -1:, :].argmax(dim=-1)
        ctx = torch.cat([ctx, next_tok], dim=1)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    # Reset context for clean measurement
    ctx = torch.randint(0, VOCAB_SIZE, (1, prompt_len), device=DEVICE)

    t0 = time.perf_counter()
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    for _ in range(n_decode):
        with torch.no_grad():
            logits = model(ctx)
        next_tok = logits[:, -1:, :].argmax(dim=-1)
        ctx = torch.cat([ctx, next_tok], dim=1)
        # Limit context to MAX_SEQ to avoid OOM
        if ctx.shape[1] >= MAX_SEQ:
            ctx = ctx[:, -MAX_SEQ:]

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return n_decode / elapsed


tps = measure_tps(model, prompt_len=64, n_decode=30)
print(f"  Decode TPS (FP32, no KV cache): {tps:.1f} tok/s")
assert tps > 0, "TPS should be positive"

# VRAM footprint
if DEVICE == "cuda":
    torch.cuda.reset_peak_memory_stats()
    idx = torch.randint(0, VOCAB_SIZE, (1, 256), device=DEVICE)
    with torch.no_grad():
        model(idx)
    vram_mb = torch.cuda.max_memory_allocated() / 1e6
    print(f"  Peak VRAM (prompt=256): {vram_mb:.0f} MB")
else:
    vram_mb = 0.0
    print(f"  Peak VRAM: N/A (CPU-only)")

print("  ✓ Section 3 passed — decode TPS and VRAM measured")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Saving the baseline snapshot
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Saving the Baseline Snapshot ──")
print("""
  The baseline snapshot is the "before" record for this capstone.
  Every subsequent optimisation will be compared against it.

  BASELINE SNAPSHOT SCHEMA:
    model_config  : architecture hyperparameters
    hardware      : device, VRAM capacity
    ttft_by_len   : TTFT mean/P99 at each prompt length
    tps_fp32      : decode TPS without optimisation
    vram_mb       : peak memory during inference
    dtype         : "float32" (will become float16 in 16.2)

  TODO 3: Save the baseline dict as /tmp/capstone16_baseline.json
  and assert the file was created successfully.
""")


def save_baseline(path: str, model_config: dict, ttft: dict,
                  tps: float, vram_mb: float) -> dict:
    """
    TODO 3: Build and save the baseline snapshot dict.
    Include model_config, hardware, ttft_by_len, tps_fp32, vram_mb, dtype.
    Write JSON to path. Return the dict.
    """
    baseline = {
        "model_config": model_config,
        "hardware": {"device": DEVICE, "dtype": "float32"},
        "ttft_by_len": ttft,
        "tps_fp32": round(tps, 1),
        "vram_mb": round(vram_mb, 1),
    }
    with open(path, "w") as f:
        json.dump(baseline, f, indent=2)
    return baseline


model_config = {
    "vocab_size": VOCAB_SIZE, "d_model": D_MODEL, "n_heads": N_HEADS,
    "n_layers": N_LAYERS, "d_ff": D_FF, "max_seq": MAX_SEQ,
    "n_params_M": round(n_params / 1e6, 1),
}
baseline_path = "/tmp/capstone16_baseline.json"
baseline = save_baseline(baseline_path, model_config, ttft_results, tps, vram_mb)

import os
assert os.path.exists(baseline_path), "Baseline file should be created"
assert baseline["tps_fp32"] > 0, "TPS should be positive"

print(f"  Baseline saved: {baseline_path}")
print(f"  TPS (FP32):   {baseline['tps_fp32']} tok/s")
print(f"  TTFT (256):   {ttft_results.get(256, {}).get('mean_ms', 'N/A')} ms")
print(f"  VRAM peak:    {baseline['vram_mb']} MB")
print(f"""
  NEXT STEP:
    Run 16.2_precision_and_compile.py to apply the optimisation ladder.
    The baseline JSON will be loaded for before/after comparison.
""")
print("  ✓ Section 4 passed — baseline snapshot saved")


print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Capstone 16.1 complete!")
print("  Baseline established: TTFT, TPS (FP32), VRAM footprint.")
print("  Next: VI.Capstone_Projects/16.LLM_Inference_Optimisation/16.2_precision_and_compile.py")
print("=" * 60)
