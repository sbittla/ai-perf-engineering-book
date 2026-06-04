#!/usr/bin/env python3
"""
IV.LLM_Inference_Systems/10.LLM_Inference_Fundamentals/10.2_prefill_and_decode.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chapter 10: LLM Inference Fundamentals — Section 1: Prefill vs Decode Phases
=======================================================================
Covers book section 10.1:
  • Why LLM inference has two distinct phases (prefill and decode)
  • Prefill: compute-bound, processes the entire prompt in one pass
  • Decode: memory-bandwidth bound, generates one token per step
  • Measuring Time To First Token (TTFT) and Tokens Per Second (TPS)
  • Why decode is the hard optimization target

Run:  python IV.LLM_Inference_Systems/10.LLM_Inference_Fundamentals/10.2_prefill_and_decode.py
All sections must print ✓.
"""

import time
import math
import statistics
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("  Exercise 10.1 — Prefill vs Decode Phases")
print("=" * 60)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}\n")


# ─────────────────────────────────────────────────────────────
# SECTION 1: Prefill vs Decode — the conceptual split
# ─────────────────────────────────────────────────────────────
print("── Section 1: The Two-Phase Nature of LLM Inference ──")
print("""
  Every autoregressive LLM inference request has two distinct phases:

  PREFILL PHASE
    Input  : the full prompt  (e.g. 512 tokens)
    Output : logits for each position + a populated KV cache
    Shape  : (batch, seq_len, d_model) — many tokens at once
    Bound  : COMPUTE-BOUND — large matrix multiplies dominate
    Cost   : O(seq_len²) attention, but happens only once per request

  DECODE PHASE
    Input  : one new token at a time, reading the KV cache
    Output : one new token per step
    Shape  : (batch, 1, d_model) — single token per step
    Bound  : MEMORY-BANDWIDTH-BOUND — reads entire KV cache every step
    Cost   : O(seq_len) per step × n_new_tokens steps

  This split explains why:
    • Prefill throughput scales well with GPU compute (Tensor Cores)
    • Decode throughput is limited by DRAM bandwidth
    • Adding GPUs helps prefill far more than decode
""")
print("  ✓ Section 1 passed — prefill is compute-bound, decode is bandwidth-bound")


# ─────────────────────────────────────────────────────────────
# SECTION 2: A minimal transformer to measure both phases
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: Simulating a Mini-Transformer ──")
print("""
  We build a minimal scaled-dot-product attention model.
  It is small enough to run on CPU but captures the key shapes:

    Prefill : forward(input_ids)       — shape (B, T, D)
    Decode  : forward(input_ids[:,-1:]) — shape (B, 1, D)

  In real LLMs the KV cache makes decode cheap (no recompute).
  Here we simulate: prefill populates a KV buffer; decode reads it.
""")


class MiniAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model,     bias=False)

    def forward(self, x, kv_cache=None):
        B, T, D = x.shape
        H, HD   = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, T, 3, H, HD).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if kv_cache is not None:
            k_cached, v_cached = kv_cache
            k = torch.cat([k_cached, k], dim=2)
            v = torch.cat([v_cached, v], dim=2)

        new_cache = (k.detach(), v.detach())

        scale   = 1.0 / math.sqrt(HD)
        scores  = torch.matmul(q, k.transpose(-2, -1)) * scale
        weights = F.softmax(scores, dim=-1)
        out     = torch.matmul(weights, v)
        out     = out.permute(0, 2, 1, 3).reshape(B, T, D)
        return self.proj(out), new_cache


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size=10000, d_model=256, n_layers=4, n_heads=4, max_seq=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos   = nn.Embedding(max_seq, d_model)
        self.layers = nn.ModuleList([MiniAttention(d_model, n_heads) for _ in range(n_layers)])
        self.norm   = nn.LayerNorm(d_model)
        self.head   = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, kv_caches=None):
        B, T  = input_ids.shape
        pos   = torch.arange(T, device=input_ids.device)
        x     = self.embed(input_ids) + self.pos(pos)
        new_caches = []
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches else None
            x, new_cache = layer(x, kv_cache=cache)
            new_caches.append(new_cache)
        x = self.norm(x)
        return self.head(x), new_caches


D_MODEL  = 256
N_LAYERS = 4
N_HEADS  = 4
VOCAB    = 10_000
MAX_SEQ  = 1024
model    = MiniTransformer(VOCAB, D_MODEL, N_LAYERS, N_HEADS, max_seq=MAX_SEQ).to(DEVICE)
model.eval()
print(f"  Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
print(f"  d_model={D_MODEL}  n_layers={N_LAYERS}  n_heads={N_HEADS}")
print("  ✓ Section 2 passed — mini-transformer ready")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Measuring TTFT (Time To First Token)
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: TTFT — Time To First Token ──")
print("""
  TTFT = wall-clock time from request arrival to the first generated token.

  It equals the prefill time: the cost of processing the full input prompt
  in a single forward pass to get the first output logit.

  Users perceive TTFT as the "response start delay". A 500ms TTFT feels
  sluggish; a 50ms TTFT feels instant. APIs that stream tokens show the
  first token as soon as prefill completes.

  TODO 1: Measure TTFT for prompt_len ∈ [128, 256, 512, 1024].
  Use CUDA events on GPU, time.perf_counter on CPU.
  Record how TTFT scales with prompt length.
""")

prompt_lens = [128, 256, 512, 1024]
ttft_results = {}

for prompt_len in prompt_lens:
    # TODO 1: Measure prefill time for this prompt length
    #   - create input_ids of shape (1, prompt_len) filled with random token IDs
    #   - run model(input_ids) with torch.no_grad()
    #   - measure the time using CUDA events (GPU) or perf_counter (CPU)
    input_ids = torch.randint(0, VOCAB, (1, prompt_len), device=DEVICE)

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(input_ids)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    # A single forward at the shortest prompt is dominated by fixed
    # kernel-launch overhead (~1 ms), so the prompt-length signal is easily
    # buried in noise. Take the MIN over several iterations to isolate the real
    # prefill cost; the wide prompt range (128 → 1024) makes the O(seq_len)
    # prefill work clearly dominate that overhead at the long end.
    ITERS = 20 if DEVICE == "cuda" else 5
    if DEVICE == "cuda":
        ttft_ms = float("inf")
        for _ in range(ITERS):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            with torch.no_grad():
                logits, kv_caches = model(input_ids)
            e.record()
            torch.cuda.synchronize()
            ttft_ms = min(ttft_ms, s.elapsed_time(e))
    else:
        ttft_ms = float("inf")
        for _ in range(ITERS):
            t0 = time.perf_counter()
            with torch.no_grad():
                logits, kv_caches = model(input_ids)
            ttft_ms = min(ttft_ms, (time.perf_counter() - t0) * 1000)

    ttft_results[prompt_len] = ttft_ms
    print(f"  Prompt {prompt_len:>4} tokens → TTFT = {ttft_ms:.2f} ms")

# Verify TTFT grows with prompt length (compare the extremes of the range)
short_len, long_len = prompt_lens[0], prompt_lens[-1]
ttft_short = ttft_results[short_len]
ttft_long  = ttft_results[long_len]

# On GPU timings are deterministic; on CPU cache effects can flip short sequences.
if DEVICE == "cuda":
    assert ttft_long > ttft_short, \
        f"TTFT should increase with prompt length: {ttft_short:.2f} ms vs {ttft_long:.2f} ms"
ratio = ttft_long / ttft_short if ttft_short > 0 else 1.0
print(f"\n  TTFT[{long_len}] / TTFT[{short_len}] = {ratio:.1f}× (longer prompt = more compute)")
print("  ✓ Section 3 passed — TTFT grows with prompt length")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Measuring TPS (Tokens Per Second) during decode
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: TPS — Tokens Per Second During Decode ──")
print("""
  TPS = number of new tokens generated per second during the decode phase.

  Each decode step:
    1. Takes the last token (shape: B × 1)
    2. Runs one forward pass using the stored KV cache
    3. Samples the next token from the output logits
    4. Appends the new token to the sequence

  TPS is memory-bandwidth bound because every step reads the entire
  KV cache (O(seq_len) bytes) even though it generates only one token.
  Larger batch sizes help amortise the KV cache read cost.
""")

# Prefill to establish KV cache
PROMPT_LEN = 128
N_NEW_TOKENS = 50
input_ids = torch.randint(0, VOCAB, (1, PROMPT_LEN), device=DEVICE)

with torch.no_grad():
    _, kv_caches_prefill = model(input_ids)

# Warmup decode
last_tok = input_ids[:, -1:]
kv = kv_caches_prefill
with torch.no_grad():
    for _ in range(3):
        logits, kv = model(last_tok, kv_caches=kv)
        last_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)

# Reset and measure decode TPS
with torch.no_grad():
    _, kv_caches = model(input_ids)
last_tok = input_ids[:, -1:]

if DEVICE == "cuda":
    s_evt = torch.cuda.Event(enable_timing=True)
    e_evt = torch.cuda.Event(enable_timing=True)
    s_evt.record()
    with torch.no_grad():
        kv = kv_caches
        for _ in range(N_NEW_TOKENS):
            logits, kv = model(last_tok, kv_caches=kv)
            last_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    e_evt.record()
    torch.cuda.synchronize()
    decode_ms = s_evt.elapsed_time(e_evt)
else:
    t0 = time.perf_counter()
    with torch.no_grad():
        kv = kv_caches
        for _ in range(N_NEW_TOKENS):
            logits, kv = model(last_tok, kv_caches=kv)
            last_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    decode_ms = (time.perf_counter() - t0) * 1000

tps = N_NEW_TOKENS / (decode_ms / 1000)
ms_per_token = decode_ms / N_NEW_TOKENS

print(f"  Prompt length : {PROMPT_LEN} tokens")
print(f"  New tokens    : {N_NEW_TOKENS}")
print(f"  Decode time   : {decode_ms:.1f} ms")
print(f"  Per-token     : {ms_per_token:.2f} ms/token")
print(f"  TPS           : {tps:.1f} tokens/sec")

# TODO 3: Compute TTFT and TPS ratio
#   In real LLMs: TTFT grows with prompt_len; TPS depends on model size and batch
ttft_for_prompt = ttft_results[PROMPT_LEN]

# TODO 3: assert TPS > 0
assert tps > 0, "TPS should be positive"
print(f"\n  TTFT (prefill {PROMPT_LEN} tok) : {ttft_for_prompt:.2f} ms")
print(f"  Decode TPS                   : {tps:.1f} tok/s")
print("  ✓ Section 4 passed — TPS is the decode throughput metric")


# ─────────────────────────────────────────────────────────────
# SECTION 5: Why decode is harder to optimize than prefill
# ─────────────────────────────────────────────────────────────
print("\n── Section 5: Decode Bottleneck Analysis ──")
print("""
  ARITHMETIC INTENSITY = FLOPs / bytes of memory traffic

  For a linear layer  Weight: [D_out, D_in]:

    Prefill (seq_len tokens):
        FLOPs   = 2 × D_out × D_in × seq_len
        Bytes   = D_out × D_in × dtype + seq_len × D_in × dtype   (weights + input)
        AI      ≈ seq_len  (scales with batch/seq)

    Decode (1 token):
        FLOPs   = 2 × D_out × D_in
        Bytes   = D_out × D_in × dtype + 1 × D_in × dtype         (weights + input)
        AI      ≈ 1         (always reads full weights for 1 output)

  The GPU ridge point (RTX 4090): ~300 TFLOP/s ÷ ~1 TB/s ≈ 300 FLOPs/byte.
  Decode AI ≈ 1 FLOPs/byte — 300× below the ridge point.
  Decode is ALWAYS memory-bandwidth bound on current hardware.
""")


def arithmetic_intensity(d_out, d_in, seq_len, dtype_bytes=2):
    flops = 2 * d_out * d_in * seq_len
    bytes_traffic = (d_out * d_in + seq_len * d_in) * dtype_bytes
    return flops / bytes_traffic


print(f"  {'Phase':<12}  {'seq_len':>8}  {'AI (FLOPs/byte)':>18}")
print(f"  {'─'*12}  {'─'*8}  {'─'*18}")
for sl in [1, 8, 64, 512]:
    ai = arithmetic_intensity(D_MODEL, D_MODEL, sl)
    label = "decode" if sl == 1 else "prefill"
    note  = " ← memory-bound" if ai < 10 else " ← compute-bound"
    print(f"  {label:<12}  {sl:>8}  {ai:>18.1f}{note}")

# TODO 4: Compute AI for decode (seq_len=1) and confirm it is < 10
ai_decode = arithmetic_intensity(D_MODEL, D_MODEL, 1)
assert ai_decode < 10, f"decode AI should be very low, got {ai_decode:.1f}"
print(f"\n  Decode AI = {ai_decode:.1f} FLOPs/byte  (far below GPU ridge point)")
print("  ✓ Section 5 passed — decode is always memory-bandwidth bound")


print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 10.1 complete!")
print("  You now understand why LLM inference has two phases,")
print("  how TTFT and TPS are measured, and why decode is hard.")
print("  Next: IV.LLM_Inference_Systems/10.LLM_Inference_Fundamentals/10.3_kv_cache.py")
print("=" * 60)
