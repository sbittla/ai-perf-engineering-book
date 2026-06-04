#!/usr/bin/env python3
"""
30.Cloud_Cost/30.1_cost_optimization.py  —  Chapter 30: Capstone 7
=======================================================================
Cloud Cost Optimization: rank A100/H100/L40S/RTX4090/MI300X by cost/token and
recommend a deployment architecture by binding constraint. Covers 30.1 - 30.3.

Difficulty: ***--  (3/5 - economics + decision logic)
Est. time:  45 minutes
Expected ranges (illustrative):
  - Cheapest tokens/$ for small models: RTX 4090 / L40S
  - Large models (70B+) require H100 / MI300X memory before cost matters
Troubleshooting:
  - All pricing/throughput are reference figures; replace with your cloud's
    actual quotes and your measured throughput for a real decision.
Challenge extension:
  - Add a mixed-fleet optimizer: given a traffic mix of small and large models,
    minimize total $/day by assigning each model class to its cheapest viable GPU.

Run:  python IX.Production_Capstones/30.Cloud_Cost/30.1_cost_optimization.py
"""
print("=" * 70)
print("  Exercise 30.1 - Capstone 7: Cloud Cost Optimization")
print("=" * 70)

# GPU reference profiles: $/hr, VRAM GB, rel. throughput on an 8B model (tok/s)
GPUS = {
    #          usd_hr  vram   tok_s_8b
    "RTX4090": (0.40,   24,    3200),
    "L40S":    (1.00,   48,    5200),
    "A100":    (2.00,   80,    6500),
    "H100":    (3.00,   80,   14000),
    "MI300X":  (2.20,  192,   13000),
}


# ─────────────────────────────────────────────────────────────
# SECTION 1: Cost matrix -> cost per million tokens
# ─────────────────────────────────────────────────────────────
print("\n-- Section 1: Accelerator Cost Matrix --")
print("""
  Rank by cost per million tokens (= $/hr / tokens-per-hour), not peak FLOPs.
""")

def cost_per_m(usd_hr, tok_s):
    return usd_hr / (tok_s * 3600) * 1_000_000

rows = []
for g, (usd, vram, tok_s) in GPUS.items():
    c = cost_per_m(usd, tok_s)
    rows.append((g, c, vram))
rows.sort(key=lambda r: r[1])
print(f"  {'GPU':<8} {'$/1M tok':>10} {'VRAM':>6}")
for g, c, vram in rows:
    print(f"  {g:<8} {c:>9.3f}  {vram:>4} GB")
cheapest = rows[0][0]
assert cheapest in ("RTX4090", "L40S"), "small-model $/tok winner is a cheap GPU"
print(f"  [check] cheapest tokens/$ on the 8B model: {cheapest}")


# ─────────────────────────────────────────────────────────────
# SECTION 2: Match workload to hardware (binding constraint)
# ─────────────────────────────────────────────────────────────
print("\n-- Section 2: Workload -> Hardware --")
print("""
  Memory is a hard gate: if the model + KV cache do not fit, throughput is zero.
  Among GPUs that fit, pick the cheapest per token.
""")

def recommend(model_gb, candidates):
    fit = [(g, cost_per_m(GPUS[g][0], GPUS[g][2])) for g in candidates
           if GPUS[g][1] >= model_gb]
    return min(fit, key=lambda x: x[1])[0] if fit else None

small = recommend(16, GPUS)         # 8B fp16 ~16 GB
large = recommend(140, GPUS)        # 70B fp16 ~140 GB
print(f"  8B model  (~16 GB): recommend {small}")
print(f"  70B model (~140 GB): recommend {large}")
assert small in ("RTX4090", "L40S")
assert large == "MI300X", "only the 192 GB part fits a 70B model on one GPU"
print("  [check] memory gate then cost decides the recommendation")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Mixed-fleet vs single-SKU
# ─────────────────────────────────────────────────────────────
print("\n-- Section 3: Mixed Fleet Beats Single SKU --")
print("""
  Serve the bulk of small-model traffic on cheap GPUs and reserve premium,
  high-memory GPUs for the large models that need them.
""")

# daily token demand
small_tokens = 5e9     # 5B tokens/day of 8B traffic
large_tokens = 2e8     # 0.2B tokens/day of 70B traffic

def daily_cost(gpu, tokens):
    return cost_per_m(GPUS[gpu][0], GPUS[gpu][2]) * tokens / 1_000_000

mixed = daily_cost(small, small_tokens) + daily_cost(large, large_tokens)
single = daily_cost("H100", small_tokens) + daily_cost("H100", large_tokens)  # H100 can't fit 70B alone, illustrative upper bound
print(f"  mixed fleet ({small} + {large}): ${mixed:,.0f}/day")
print(f"  single SKU (H100 only):         ${single:,.0f}/day")
assert mixed < single, "mixed fleet should be cheaper than premium single-SKU"
print("  [check] mixed fleet lowers total daily cost")


print("\n" + "=" * 70)
print("  ALL SECTIONS PASSED - Exercise 30.1 complete")
print("  You ranked GPUs by cost/token, matched workloads to hardware by the")
print("  memory gate, and showed a mixed fleet beats a single premium SKU.")
print("=" * 70)
