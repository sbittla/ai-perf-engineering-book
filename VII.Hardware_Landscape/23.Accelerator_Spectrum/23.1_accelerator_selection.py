#!/usr/bin/env python3
"""
23.Accelerator_Spectrum/23.1_accelerator_selection.py  —  Chapter 23: The Accelerator Spectrum
=======================================================================
Covers book sections 23.1 - 23.5:
  23.1  The accelerator landscape and the five decision axes
  23.2  ASICs - where fixed-function silicon wins
  23.3  FPGAs - the low-latency reconfigurable niche
  23.4  Edge AI - NPUs, quantization, and the power envelope
  23.5  Choosing an accelerator - a decision framework

Run:
    python VII.Hardware_Landscape/23.Accelerator_Spectrum/23.1_accelerator_selection.py

No GPU required - all sections use reference data tables. All sections print a check.
"""

print("=" * 70)
print("  Exercise 23.1 - The Accelerator Spectrum: Selection Calculator")
print("=" * 70)


# Reference accelerator profiles (illustrative, order-of-magnitude figures).
# tps   = relative throughput (inferences/s, normalized to GPU = 1.0)
# watts = typical board power under load
# usd   = approximate unit acquisition cost
ACCELERATORS = {
    #            tps    watts    usd       flexibility (1=fixed .. 5=fully programmable)
    "GPU":      (1.00,  700,    30000,    5),
    "ASIC":     (1.30,  400,    12000,    1),
    "FPGA":     (0.25,  225,    10000,    3),
    "Edge-NPU": (0.02,  10,       150,    2),
}


# ─────────────────────────────────────────────────────────────
# SECTION 1: The five decision axes
# ─────────────────────────────────────────────────────────────
print("\n-- Section 1: Throughput-per-Watt and Throughput-per-Dollar --")
print("""
  The accelerator choice is a performance-engineering question with five
  axes: throughput, latency, performance-per-watt, performance-per-dollar,
  and flexibility (toolchain maturity / tolerance for model churn).

  Here we compute the two efficiency ratios for each reference profile.
""")

print(f"  {'Accelerator':<12} {'tps':>6} {'watts':>7} {'tps/W':>9} {'tps/$':>11}")
print(f"  {'-'*12} {'-'*6} {'-'*7} {'-'*9} {'-'*11}")
metrics = {}
for name, (tps, watts, usd, flex) in ACCELERATORS.items():
    tps_per_w = tps / watts
    tps_per_d = tps / usd
    metrics[name] = (tps_per_w, tps_per_d)
    print(f"  {name:<12} {tps:>6.2f} {watts:>7} {tps_per_w:>9.4f} {tps_per_d:>11.6f}")

# The ASIC has the best throughput-per-watt (it is built for exactly this).
# The Edge-NPU does NOT win tps/W - its edge is the lowest ABSOLUTE power, i.e.
# it is the only part that fits inside a single-digit-watt budget.
assert metrics["ASIC"][0] > metrics["GPU"][0], "ASIC should beat GPU on tps/W"
assert metrics["ASIC"][0] == max(m[0] for m in metrics.values()), \
    "ASIC should have the highest tps/W"
lowest_power = min(ACCELERATORS, key=lambda k: ACCELERATORS[k][1])
assert lowest_power == "Edge-NPU", "Edge-NPU should have the lowest absolute power"
print("\n  [check] ASIC wins perf/watt; Edge-NPU wins absolute power budget")


# ─────────────────────────────────────────────────────────────
# SECTION 2: ASIC vs GPU energy break-even
# ─────────────────────────────────────────────────────────────
print("\n-- Section 2: ASIC-vs-GPU Energy Break-Even --")
print("""
  An ASIC carries large non-recurring engineering (NRE) cost (a tapeout is
  tens to hundreds of millions of dollars). It only pays back when its
  perf-per-watt energy savings, multiplied by deployment volume and service
  lifetime, exceed that NRE. This is the single most important number in the
  GPU-vs-ASIC decision.
""")

def asic_breakeven_years(nre_usd, units, watts_gpu, watts_asic,
                         price_kwh=0.12, hours_per_day=20):
    """Years for ASIC energy savings to repay its NRE (lower is better)."""
    w_saved = max(watts_gpu - watts_asic, 0)
    kwh_year = w_saved / 1000 * hours_per_day * 365
    save_year = kwh_year * price_kwh * units
    return nre_usd / save_year if save_year else float("inf")

scenarios = [
    ("Hyperscaler (50k units, $50M NRE)", 50_000_000, 50_000),
    ("Mid-size fleet (5k units, $50M NRE)", 50_000_000, 5_000),
    ("Small fleet  (500 units, $50M NRE)", 50_000_000,   500),
]
print(f"  {'Scenario':<38} {'break-even (yr)':>16}")
print(f"  {'-'*38} {'-'*16}")
results = []
for label, nre, units in scenarios:
    yrs = asic_breakeven_years(nre, units, 700, 400)
    results.append(yrs)
    shown = f"{yrs:>16.2f}" if yrs != float("inf") else f"{'never':>16}"
    print(f"  {label:<38} {shown}")

# More units => faster payback (monotonic).
assert results[0] < results[1] < results[2], "more units should pay back faster"
print("\n  [check] break-even falls as deployment volume rises")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Constraint-driven recommendation
# ─────────────────────────────────────────────────────────────
print("\n-- Section 3: Matching the Binding Constraint to the Hardware --")
print("""
  The five axes collapse into a short decision procedure keyed on the single
  binding constraint of the deployment.
""")

DECISION = {
    "model_churn":        ("GPU",      "Flexibility and mature toolchain dominate"),
    "stable_high_volume": ("ASIC",     "Best perf/watt amortizes the NRE"),
    "deterministic_us":   ("FPGA",     "No kernel-launch or batching overhead"),
    "battery_thermal":    ("Edge-NPU", "Only option inside the power budget"),
}

def recommend(constraint):
    return DECISION.get(constraint, ("GPU", "Default: flexible and well-supported"))

print(f"  {'Binding constraint':<22} {'Choose':<10} Why")
print(f"  {'-'*22} {'-'*10} {'-'*40}")
for c, (hw, why) in DECISION.items():
    print(f"  {c:<22} {hw:<10} {why}")

assert recommend("battery_thermal")[0] == "Edge-NPU"
assert recommend("stable_high_volume")[0] == "ASIC"
assert recommend("deterministic_us")[0] == "FPGA"
print("\n  [check] recommendation engine maps every constraint correctly")


print("\n" + "=" * 70)
print("  ALL SECTIONS PASSED - Exercise 23.1 complete")
print("  You computed perf/watt and perf/$ across the accelerator spectrum,")
print("  modeled the ASIC-vs-GPU energy break-even, and built a")
print("  constraint-driven accelerator recommendation.")
print("=" * 70)
