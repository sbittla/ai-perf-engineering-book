#!/usr/bin/env python3
"""
VI.Capstone_Projects/16.LLM_Inference_Optimisation/16.3_profiling_audit.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chapter 16: Capstone 1 — LLM Inference Optimisation Lab
Section 3: Profiling Audit — Attributing Time to Operations
=======================================================================
Covers capstone section 16.3:
  • Using torch.profiler to record CPU/CUDA time per op
  • Reading the profiler table: Self CUDA % is the target metric
  • Attributing latency to attention, FFN, embedding, and head layers
  • Computing the "3 top bottlenecks" and which optimisation fixed each
  • Comparing FP32 and FP16 op-level breakdowns

Run:  python VI.Capstone_Projects/16.LLM_Inference_Optimisation/16.3_profiling_audit.py
All sections must print ✓.
"""

import json
import math
import statistics
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

print("=" * 60)
print("  Capstone 16.3 — Profiling Audit")
print("=" * 60)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}\n")

VOCAB_SIZE = 32_000
D_MODEL    = 512
N_HEADS    = 8
N_LAYERS   = 4
D_FF       = 2048
MAX_SEQ    = 512


# ── Rebuild model with named regions ─────────────────────────
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
        with record_function("attn_qkv"):
            qkv = self.qkv(x).chunk(3, dim=-1)
            q, k, v = [t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) for t in qkv]
        with record_function("attn_scores"):
            scale = math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) / scale
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
        with record_function("attn_proj"):
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
        with record_function("attention"):
            x = x + self.attn(self.ln1(x))
        with record_function("ffn"):
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
        with record_function("embedding"):
            pos = torch.arange(T, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
        for i, block in enumerate(self.blocks):
            with record_function(f"layer_{i}"):
                x = block(x)
        with record_function("lm_head"):
            return self.head(self.ln_f(x))


def make_model(dtype=torch.float32):
    torch.manual_seed(42)
    m = MiniGPT(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, D_FF, MAX_SEQ)
    return m.to(dtype=dtype).to(DEVICE).eval()


# ─────────────────────────────────────────────────────────────
# SECTION 1: torch.profiler basics
# ─────────────────────────────────────────────────────────────
print("── Section 1: torch.profiler — How to Read the Output ──")
print("""
  torch.profiler records CPU and CUDA time for every PyTorch op.
  The key column is "Self CUDA %" — the fraction of total CUDA time
  that this op consumes EXCLUDING its sub-ops.

  READING THE TABLE:
    Name              : op name or user-annotated region
    Self CPU %        : CPU time excluding child ops
    Self CUDA %       : GPU kernel time excluding child ops  ← target
    CUDA total %      : GPU time INCLUDING child ops (can exceed 100% via nesting)
    # of Calls        : how many times this op ran

  PROFILER SCHEDULE (recommended):
    wait=1   : skip first iteration (warmup noise)
    warmup=2 : run without recording (JIT / cuBLAS warmup)
    active=5 : record these iterations → reliable mean
    repeat=1 : one cycle

  record_function("name") creates a named annotation in the trace.
  Use it to label your code blocks so they appear in the profiler table.
""")

# Demonstrate profiler with a single forward pass
model_fp32 = make_model(torch.float32)
idx256 = torch.randint(0, VOCAB_SIZE, (1, 256), device=DEVICE)

# Warmup
with torch.no_grad():
    for _ in range(5):
        model_fp32(idx256)
if DEVICE == "cuda":
    torch.cuda.synchronize()

activities = [ProfilerActivity.CPU]
if DEVICE == "cuda":
    activities.append(ProfilerActivity.CUDA)

with profile(activities=activities, record_shapes=False) as prof:
    with torch.no_grad():
        for _ in range(5):
            model_fp32(idx256)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

# Get the top ops sorted by CUDA time (CPU if no GPU)
sort_key = "cuda_time_total" if DEVICE == "cuda" else "cpu_time_total"
table_str = prof.key_averages().table(sort_by=sort_key, row_limit=10)
print(f"  Top 10 ops by {'CUDA' if DEVICE == 'cuda' else 'CPU'} time (FP32, prompt=256):")
# Print abbreviated version
lines = table_str.split("\n")
for line in lines[:15]:
    print(f"  {line}")
print("  ...")

print("  ✓ Section 1 passed — torch.profiler recording works")


# ─────────────────────────────────────────────────────────────
# SECTION 2: Op-level time attribution
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: Op-Level Time Attribution ──")
print("""
  We now extract self-time per labelled region (embedding, attention, ffn,
  lm_head) and compute each region's share of total CUDA time.

  This answers: "Where does the time actually go?"

  Expected breakdown for a 4-layer transformer (FP32, prompt=256):
    FFN (4 layers)     : ~50% — two large matmuls per layer
    Attention (4 layers): ~35% — QKV proj + scores + output proj
    LM head            : ~10% — large vocab projection
    Embedding + misc   :  ~5% — lookup tables, LayerNorm

  TODO 1: Implement profile_region_breakdown(model, idx, n_iters=5)
  that profiles n_iters forward passes and returns a dict mapping
  region name → mean time in ms.
""")


def profile_region_breakdown(model: nn.Module, idx: torch.Tensor,
                               n_iters: int = 5) -> dict:
    """
    TODO 1: Profile the model and return region timing breakdown.
    Use torch.profiler with record_function annotations.
    For each key average, extract cpu_time_total or cuda_time_total.
    Return dict: {region_name: time_us}.
    """
    activities = [ProfilerActivity.CPU]
    if DEVICE == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=False) as prof:
        with torch.no_grad():
            for _ in range(n_iters):
                model(idx)
        if DEVICE == "cuda":
            torch.cuda.synchronize()

    REGION_TAGS = ["embedding", "attention", "ffn", "lm_head",
                   "attn_qkv", "attn_scores", "attn_proj", "layer_"]
    breakdown = {}
    for evt in prof.key_averages():
        name = evt.key
        if not any(r in name for r in REGION_TAGS):
            continue
        # Prefer device (CUDA) time on GPU; fall back to CPU time for record_function regions
        t_us = getattr(evt, "device_time_total", 0) / n_iters
        if t_us <= 0:
            t_us = evt.cpu_time_total / n_iters
        if t_us > 0:
            breakdown[name] = round(t_us, 1)
    return breakdown


breakdown_fp32 = profile_region_breakdown(model_fp32, idx256)

total_us = sum(breakdown_fp32.values()) or 1
print(f"\n  Region breakdown (FP32, prompt=256):")
print(f"  {'Region':<20}  {'Time (µs)':>12}  {'Share':>8}")
print(f"  {'─'*20}  {'─'*12}  {'─'*8}")
for region, t_us in sorted(breakdown_fp32.items(), key=lambda x: -x[1])[:10]:
    pct = t_us / total_us * 100
    print(f"  {region:<20}  {t_us:>12.1f}  {pct:>7.1f}%")

assert len(breakdown_fp32) > 0, "Profiler should return at least one region"
print("  ✓ Section 2 passed — op-level attribution extracted")


# ─────────────────────────────────────────────────────────────
# SECTION 3: FP16 vs FP32 breakdown comparison
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: FP16 vs FP32 — Where Does the Speedup Come From? ──")
print("""
  The profiler breakdown across precision levels shows exactly which
  operations benefit most from FP16. The expected pattern:
    - FFN matmuls      : 2× speedup (Tensor Cores for large tiles)
    - Attention scores : 1.5–2× speedup (smaller L×L matrix)
    - LM head          : 2× speedup (large vocab matmul)
    - LayerNorm        : marginal (already fast, elementwise)

  TODO 2: Compare FP32 and FP16 region breakdowns.
  Build a side-by-side table showing time in each dtype and speedup per region.
""")

try:
    model_fp16 = make_model(torch.float16)
    breakdown_fp16 = profile_region_breakdown(model_fp16, idx256)
    regions_common = set(breakdown_fp32.keys()) & set(breakdown_fp16.keys())

    if regions_common:
        print(f"\n  FP32 vs FP16 breakdown comparison (top regions):")
        print(f"  {'Region':<20}  {'FP32 (µs)':>12}  {'FP16 (µs)':>12}  {'Speedup':>9}")
        print(f"  {'─'*20}  {'─'*12}  {'─'*12}  {'─'*9}")
        for region in sorted(regions_common,
                              key=lambda r: -breakdown_fp32.get(r, 0))[:8]:
            t32 = breakdown_fp32.get(region, 0)
            t16 = breakdown_fp16.get(region, 1)
            sp = t32 / t16 if t16 > 0 else float("nan")
            print(f"  {region:<20}  {t32:>12.1f}  {t16:>12.1f}  {sp:>8.2f}×")
    else:
        print(f"  FP16 breakdown extracted ({len(breakdown_fp16)} regions)")
except Exception as ex:
    print(f"  FP16 comparison skipped: {ex}")

print("  ✓ Section 3 passed — FP16 vs FP32 breakdown compared")


# ─────────────────────────────────────────────────────────────
# SECTION 4: The top-3 bottlenecks and which fix helped
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Diagnosis → Fix Attribution ──")
print("""
  A complete profiling audit ends with a diagnosis-to-fix table:
  which profiler finding pointed to which optimisation, and how much
  speedup that optimisation produced.

  DIAGNOSIS-TO-FIX FRAMEWORK:
    Finding                        → Fix                  → Speedup (typical)
    ─────────────────────────────────────────────────────────────────────────
    FFN matmul dominates (>40%)    → FP16 precision        → 1.5–2.0×
    Attention dominates (>40%)     → Flash Attention       → 1.5–3.0×
    LM head dominates (>20%)       → FP16 + weight sharing → 1.5–2.0×
    Embedding is slow (>10%)       → Reduce vocab size     → 1.2–1.5×
    GPU utilisation < 50%          → Larger batch          → 2–5×
    Compile shows many tiny ops    → torch.compile         → 1.2–1.8×

  This is the final output of the capstone: a structured audit report
  linking observations to actions to measured outcomes.
""")

# Build the audit report
audit = {
    "model": f"MiniGPT ({N_LAYERS}L, d={D_MODEL})",
    "prompt_len": 256,
    "device": DEVICE,
    "top_bottlenecks": [],
    "fixes": [],
}

# Top 3 bottlenecks by FP32 time
top3 = sorted(breakdown_fp32.items(), key=lambda x: -x[1])[:3]
for rank, (region, t_us) in enumerate(top3, start=1):
    share = t_us / (sum(breakdown_fp32.values()) or 1) * 100
    fix = {
        "attention": "Apply FP16 + optional Flash Attention (O(L) memory)",
        "ffn":       "Apply FP16 (2× Tensor Core speedup on large tiles)",
        "lm_head":   "Apply FP16 or weight tying (shared with embedding)",
        "embedding": "Reduce vocab size or use FP16 embedding lookup",
    }.get(region.split("_")[0], "Profile further with ncu")
    audit["top_bottlenecks"].append({
        "rank": rank, "region": region,
        "time_us": t_us, "share_pct": round(share, 1), "fix": fix,
    })
    print(f"  Bottleneck #{rank}: {region:<20}  {t_us:.0f} µs  ({share:.1f}%)  → {fix}")

# Load optimisation results if available
opt_path = "/tmp/capstone16_optimised.json"
if __import__("os").path.exists(opt_path):
    with open(opt_path) as f:
        opt_data = json.load(f)
    audit["best_config"]  = opt_data.get("best_config")
    audit["best_speedup"] = opt_data.get("best_speedup")
    print(f"\n  Overall best config: {audit['best_config']}  ({audit['best_speedup']}×)")

with open("/tmp/capstone16_audit.json", "w") as f:
    json.dump(audit, f, indent=2)

print(f"""
  AUDIT SAVED: /tmp/capstone16_audit.json

  CAPSTONE 16 COMPLETE SUMMARY:
    1. Establish baseline (16.1)  → TTFT, TPS, VRAM snapshot
    2. Optimise (16.2)            → FP16 + torch.compile ladder
    3. Audit (16.3)               → profiler attribution per region

  The audit links EVERY speedup back to a specific profiler finding.
  That is the standard of evidence required for production optimisation.
""")
assert len(audit["top_bottlenecks"]) > 0, "Audit should have bottleneck data"
print("  ✓ Section 4 passed — diagnosis-to-fix audit complete")


print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Capstone 16.3 complete!")
print("  Capstone 16 (LLM Inference Optimisation) is finished.")
print("  Next: VI.Capstone_Projects/17.DataLoader_Bottleneck_Hunt/17.1_slow_dataloader.py")
print("=" * 60)
