#!/usr/bin/env python3
"""
Appendices/E.Advanced_Capstones/E.3_nccl_tuning.py  --  Appendix E.3
=======================================================================
Multi-Node NCCL Tuning: model ring vs tree AllReduce so you can read an
nccl-tests bandwidth curve and find the scaling cliff before you tune it.

Difficulty: *****  (5/5 - Expert; real cluster is the live extension)
Est. time:  1-2 days on a 2+ node cluster
Expected ranges:
  - Tree wins at small messages (latency-bound); ring wins at large (bw-bound)
  - Ring bus bandwidth approaches link bandwidth as message size grows
Troubleshooting:
  - No cluster needed here; this models the curves you will measure.
Live run (the real capstone):
  - Measure with: all_reduce_perf -b 1K -e 1G -f 2 -g <gpus>
  - Tune NCCL_ALGO / buffer sizes / topology; tabulate efficiency 1->16 GPUs.

Run:  python Appendices/E.Advanced_Capstones/E.3_nccl_tuning.py
"""
print("=" * 70)
print("  Exercise E.3 - Multi-Node NCCL Tuning (ring vs tree model)")
print("=" * 70)

N = 16                  # total GPUs across nodes
LINK_GBs = 50.0         # per-link inter-node bandwidth (e.g. InfiniBand)
LATENCY_US = 5.0        # per-hop latency

# ---------------------------------------------------------------------------
# SECTION 1: Ring AllReduce bus bandwidth vs message size
# ---------------------------------------------------------------------------
print("\n-- Section 1: Ring AllReduce Bandwidth --")
print("""
  Ring AllReduce moves 2*(N-1)/N * S bytes per GPU and takes 2*(N-1) steps.
  Small messages are dominated by per-step latency; large messages approach
  the link's streaming bandwidth (the 'bus bandwidth').
""")

def ring_time_us(S_bytes, n, link_gbs, lat_us):
    steps = 2 * (n - 1)
    lat = steps * lat_us
    moved = 2 * (n - 1) / n * S_bytes
    bw = moved / (link_gbs * 1e9) * 1e6     # us
    return lat + bw

def ring_busbw_gbs(S_bytes, n, link_gbs, lat_us):
    moved = 2 * (n - 1) / n * S_bytes
    t_s = ring_time_us(S_bytes, n, link_gbs, lat_us) / 1e6
    return moved / t_s / 1e9

print(f"  {'msg':>8} {'ring us':>10} {'bus GB/s':>10}")
for S in (1024, 64*1024, 1024*1024, 64*1024*1024, 1024*1024*1024):
    t = ring_time_us(S, N, LINK_GBs, LATENCY_US)
    busbw = ring_busbw_gbs(S, N, LINK_GBs, LATENCY_US)
    label = f"{S//1024}K" if S < 1024*1024 else f"{S//(1024*1024)}M"
    print(f"  {label:>8} {t:>10.1f} {busbw:>10.2f}")
big_bw = ring_busbw_gbs(1024*1024*1024, N, LINK_GBs, LATENCY_US)
assert big_bw > 0.6 * LINK_GBs
print("  [check] large messages approach link bandwidth (bw-bound)")

# ---------------------------------------------------------------------------
# SECTION 2: Tree vs ring - latency scaling
# ---------------------------------------------------------------------------
print("\n-- Section 2: Tree vs Ring --")
print("""
  A tree AllReduce completes in ~2*log2(N) steps instead of 2*(N-1). For small,
  latency-bound messages that is a large win; for big messages ring's higher
  bandwidth utilization wins. NCCL_ALGO lets you force one or the other.
""")
import math

def tree_time_us(S_bytes, n, link_gbs, lat_us):
    steps = 2 * max(1, int(math.ceil(math.log2(n))))
    lat = steps * lat_us
    # tree moves roughly S up and S down the tree per GPU
    bw = (2 * S_bytes) / (link_gbs * 1e9) * 1e6
    return lat + bw

print(f"  {'msg':>8} {'ring us':>10} {'tree us':>10} {'winner':>8}")
crossover = None
for S in (1024, 8*1024, 64*1024, 512*1024, 4*1024*1024, 64*1024*1024):
    r = ring_time_us(S, N, LINK_GBs, LATENCY_US)
    t = tree_time_us(S, N, LINK_GBs, LATENCY_US)
    win = "tree" if t < r else "ring"
    if crossover is None and win == "ring":
        crossover = S
    label = f"{S//1024}K" if S < 1024*1024 else f"{S//(1024*1024)}M"
    print(f"  {label:>8} {r:>10.1f} {t:>10.1f} {win:>8}")
assert tree_time_us(1024, N, LINK_GBs, LATENCY_US) < ring_time_us(1024, N, LINK_GBs, LATENCY_US)
assert ring_time_us(64*1024*1024, N, LINK_GBs, LATENCY_US) < tree_time_us(64*1024*1024, N, LINK_GBs, LATENCY_US)
print(f"  Crossover near        : ~{crossover//1024}K bytes")
print("  [check] tree wins small (latency), ring wins large (bandwidth)")

# ---------------------------------------------------------------------------
# SECTION 3: The scaling cliff at the node boundary
# ---------------------------------------------------------------------------
print("\n-- Section 3: The Node-Boundary Cliff --")
print("""
  Inside a node, NVLink (~600 GB/s) carries the collective; once it crosses to a
  second node over InfiniBand (~50 GB/s) the same AllReduce gets ~10x slower.
  That step-down is the efficiency cliff you see going from 8 to 16 GPUs.
""")
S = 1024*1024*1024
intra = ring_time_us(S, 8, 600.0, 2.0)
inter = ring_time_us(S, 16, 50.0, 5.0)
print(f"  8 GPU  (NVLink 600 GB/s) : {intra/1000:7.2f} ms")
print(f"  16 GPU (IB     50 GB/s)  : {inter/1000:7.2f} ms  ({inter/intra:.0f}x slower)")
assert inter > intra * 5
print("  [check] the node boundary, not GPU count, causes the cliff")

print("\n" + "=" * 70)
print("  ALL SECTIONS PASSED - Exercise E.3 complete")
print("  You modeled ring vs tree and located the node-boundary cliff. Now")
print("  measure it with nccl-tests and tune NCCL_ALGO / topology to recover it.")
print("=" * 70)
