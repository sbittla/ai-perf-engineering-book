#!/usr/bin/env python3
"""
Appendices/E.Advanced_Capstones/E.4_quant_bakeoff.py  --  Appendix E.4
=======================================================================
Quantization Bake-Off: build the accuracy / latency / memory Pareto frontier
across FP16, FP8, INT8 (W8A8), and INT4, then pick a precision by SLA.

Difficulty: ****-  (4/5 - analytic; real measurement is the live extension)
Est. time:  8-12 hours to run all variants on real hardware
Expected ranges:
  - Memory shrinks monotonically FP16 > FP8 ~ INT8 > INT4
  - Decode latency (bandwidth-bound) tracks bytes read per token
  - Accuracy drop grows as bit-width falls; INT4 needs AWQ/GPTQ to stay usable
Troubleshooting:
  - No GPU needed here; replace the reference deltas with your measured numbers.
Live run (the real capstone):
  - Produce each variant, measure a task metric + TTFT/TPS + peak VRAM, and plot
    the Pareto frontier. Recommend a precision per SLA regime.

Run:  python Appendices/E.Advanced_Capstones/E.4_quant_bakeoff.py
"""
print("=" * 70)
print("  Exercise E.4 - Quantization Bake-Off (accuracy/latency/memory Pareto)")
print("=" * 70)

PARAMS = 7e9        # 7B model
# (name, bits_per_weight, relative accuracy drop %, activations also quantized?)
VARIANTS = [
    ("FP16",        16, 0.0,  False),
    ("FP8",          8, 0.3,  True),
    ("INT8 (W8A8)",  8, 0.8,  True),
    ("INT4 (AWQ)",   4, 2.0,  False),
]

# ---------------------------------------------------------------------------
# SECTION 1: Memory footprint
# ---------------------------------------------------------------------------
print("\n-- Section 1: Weight Memory --")
print("""
  Weight memory = params * bits / 8. This sets how big a model fits and how many
  bytes the bandwidth-bound decode step must read per token.
""")
mem = {}
print(f"  {'variant':>14} {'GB':>7}")
for name, bits, _, _ in VARIANTS:
    gb = PARAMS * bits / 8 / 1e9
    mem[name] = gb
    print(f"  {name:>14} {gb:>7.2f}")
assert mem["FP16"] > mem["FP8"] >= mem["INT4 (AWQ)"]
print("  [check] memory shrinks monotonically as bit-width falls")

# ---------------------------------------------------------------------------
# SECTION 2: Decode latency (bandwidth-bound) tracks bytes/token
# ---------------------------------------------------------------------------
print("\n-- Section 2: Decode Latency --")
print("""
  Decode reads the full weight set per token, so per-token time ~ weight bytes /
  bandwidth. Lower precision => fewer bytes => proportionally faster decode.
""")
BW_GBs = 270.0
lat = {}
print(f"  {'variant':>14} {'ms/token':>10} {'speedup':>9}")
base = None
for name, bits, _, _ in VARIANTS:
    ms = (PARAMS * bits / 8) / (BW_GBs * 1e9) * 1000
    lat[name] = ms
    base = base or ms
    print(f"  {name:>14} {ms:>10.2f} {base/ms:>8.2f}x")
assert lat["INT4 (AWQ)"] < lat["FP16"]
print("  [check] lower precision speeds up bandwidth-bound decode")

# ---------------------------------------------------------------------------
# SECTION 3: Pareto frontier and SLA-driven choice
# ---------------------------------------------------------------------------
print("\n-- Section 3: Pareto Frontier + SLA Pick --")
print("""
  Plot accuracy-drop (lower better) against latency (lower better). A variant is
  on the frontier if nothing else is better on BOTH axes. Pick by the binding
  SLA: max acceptable accuracy drop and max acceptable latency.
""")

def on_frontier(name):
    a = dict((n, d) for n, _, d, _ in VARIANTS)[name]
    l = lat[name]
    for n2, _, d2, _ in VARIANTS:
        if n2 == name:
            continue
        if d2 <= a and lat[n2] <= l and (d2 < a or lat[n2] < l):
            return False
    return True

frontier = [n for n, _, _, _ in VARIANTS if on_frontier(n)]
print("  Pareto-optimal variants:", ", ".join(frontier))
assert "FP16" in frontier and "INT4 (AWQ)" in frontier

def pick(max_acc_drop, max_ms):
    cands = [(n, d) for n, _, d, _ in VARIANTS if d <= max_acc_drop and lat[n] <= max_ms]
    # cheapest memory among those meeting the SLA
    return min(cands, key=lambda x: mem[x[0]])[0] if cands else None

slas = [("tight quality, loose latency", 0.5, 100.0),
        ("loose quality, tight latency", 5.0, lat["INT4 (AWQ)"] + 0.01)]
for label, acc, ms in slas:
    print(f"  SLA [{label:30}] -> {pick(acc, ms)}")
assert pick(0.5, 100.0) in ("FP16", "FP8")
assert pick(5.0, lat["INT4 (AWQ)"] + 0.01) == "INT4 (AWQ)"
print("  [check] SLA selects the right precision from the frontier")

print("\n" + "=" * 70)
print("  ALL SECTIONS PASSED - Exercise E.4 complete")
print("  You built the precision Pareto frontier and chose by SLA. Now measure")
print("  real accuracy/latency/VRAM for each variant and redraw it with your data.")
print("=" * 70)
