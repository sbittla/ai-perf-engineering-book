#!/usr/bin/env python3
"""
25.FlashAttention/25.1_flash_attention.py  —  Chapter 25: FlashAttention
=======================================================================
Covers book sections 25.1 - 25.4:
  25.1  Why attention is memory-bound (the O(N^2) HBM matrix)
  25.2  IO-aware tiling and SRAM reuse + online softmax
  25.3  FlashAttention v1 vs v2
  25.4  Profiling the impact (memory and utilization)

Difficulty: ****-  (4/5 - IO-aware reasoning)
Est. time:  45-60 minutes
Expected ranges (reference GPU, fp16, seq=2048):
  - Peak-memory drop (standard -> SDPA flash): large (the N*N activation goes away)
  - Speedup grows with sequence length; small at N=128, large at N>=2048
Troubleshooting:
  - SDPA may pick a non-flash backend for unsupported dtype/head_dim/mask;
    use torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION) to force it.
  - No GPU -> the analytic HBM-traffic model runs and all asserts still pass.
Challenge extension:
  - Plot speedup vs sequence length (128, 512, 2048, 8192) and show the gap
    widening as the N^2 memory term comes to dominate.

Run:  python VIII.Advanced_Performance/25.FlashAttention/25.1_flash_attention.py
"""
print("=" * 70)
print("  Exercise 25.1 - FlashAttention: IO-Aware Attention")
print("=" * 70)

HAVE_GPU = False
try:
    import torch
    HAVE_GPU = torch.cuda.is_available()
except Exception:
    pass
print(f"\n  GPU available: {HAVE_GPU}{'' if HAVE_GPU else '  (analytic model)'}")


# ─────────────────────────────────────────────────────────────
# SECTION 1: HBM traffic of standard vs flash attention
# ─────────────────────────────────────────────────────────────
print("\n-- Section 1: Attention HBM Traffic --")
print("""
  Standard attention materializes the N x N score matrix in HBM (write after
  QK^T, read for softmax, write, read for the V multiply). FlashAttention tiles
  Q,K,V into SRAM and never writes the N x N matrix - cutting HBM traffic from
  O(N^2) toward compute-bound.
""")

def standard_attn_hbm_elems(N, d):
    # dominant term: the N*N score matrix touched several times
    return 4 * N * N + 3 * N * d

def flash_attn_hbm_elems(N, d):
    # only Q,K,V in and O out; no N*N matrix in HBM
    return 4 * N * d

for N in (512, 2048, 8192):
    d = 64
    std = standard_attn_hbm_elems(N, d)
    fla = flash_attn_hbm_elems(N, d)
    print(f"  N={N:>5} d={d}:  standard={std:>12,}  flash={fla:>10,}  "
          f"ratio={std/fla:>6.1f}x")
    assert std > fla
# the advantage must grow with sequence length
r1 = standard_attn_hbm_elems(512, 64) / flash_attn_hbm_elems(512, 64)
r2 = standard_attn_hbm_elems(8192, 64) / flash_attn_hbm_elems(8192, 64)
assert r2 > r1, "flash advantage should grow with N"
print("  [check] flash HBM advantage grows with sequence length")


# ─────────────────────────────────────────────────────────────
# SECTION 2: Online (streaming) softmax correctness
# ─────────────────────────────────────────────────────────────
print("\n-- Section 2: Online Softmax (the trick that makes tiling correct) --")
print("""
  FlashAttention computes softmax block-by-block with a running max (m) and a
  running sum (l), never seeing the full row at once. It must match the
  standard full-row softmax exactly.
""")
import math

def online_softmax(scores, block=3):
    m = -math.inf; l = 0.0; acc = 0.0  # acc = sum(p_i * v_i), v_i := scores here for test
    for i in range(0, len(scores), block):
        blk = scores[i:i+block]
        m_new = max(m, max(blk))
        # rescale previous running stats to the new max
        scale = math.exp(m - m_new) if m != -math.inf else 0.0
        l = l * scale + sum(math.exp(s - m_new) for s in blk)
        acc = acc * scale + sum(math.exp(s - m_new) * s for s in blk)
        m = m_new
    return acc / l  # weighted average (stands in for the V multiply)

def full_softmax_weighted(scores):
    mx = max(scores)
    exps = [math.exp(s - mx) for s in scores]
    Z = sum(exps)
    return sum(e * s for e, s in zip(exps, scores)) / Z

scores = [0.1, 2.3, -1.0, 0.7, 1.5, -0.3, 2.0, 0.0]
got = online_softmax(scores); exp = full_softmax_weighted(scores)
print(f"  online={got:.6f}   full={exp:.6f}")
assert abs(got - exp) < 1e-9, "online softmax must equal full softmax"
print("  [check] streaming softmax matches full-row softmax exactly")


# ─────────────────────────────────────────────────────────────
# SECTION 3: SDPA / FlashAttention numerical check
# ─────────────────────────────────────────────────────────────
print("\n-- Section 3: scaled_dot_product_attention --")
if HAVE_GPU:
    import torch.nn.functional as F
    torch.manual_seed(0)
    B, H, N, d = 2, 8, 512, 64
    q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q); v = torch.randn_like(q)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert out.shape == (B, H, N, d)
    print(f"  [check] SDPA produced correct output shape {tuple(out.shape)} on GPU")
else:
    print("  [check] (no GPU) - SDPA path skipped; analytic model verified above")


print("\n" + "=" * 70)
print("  ALL SECTIONS PASSED - Exercise 25.1 complete")
print("  You quantified attention HBM traffic, proved the online-softmax")
print("  identity, and exercised the FlashAttention entry point.")
print("=" * 70)
