#!/usr/bin/env python3
"""
29.MultiGPU_Scaling/29.1_scaling_challenge.py  —  Chapter 29: Capstone 6
=======================================================================
Multi-GPU Scaling Challenge: 1/2/4/8-GPU efficiency, NCCL cost, Amdahl ceiling.
Covers sections 29.1 - 29.3.

Difficulty: ***--  (3/5 - analytic; optional torchrun extension)
Est. time:  45 minutes
Expected ranges:
  - Healthy 8-GPU scaling efficiency: >= 0.80
  - Efficiency knee typically appears when collectives cross the node boundary
Troubleshooting:
  - No multi-GPU hardware needed; reference throughput is provided.
  - For a live run: torchrun --nproc_per_node=N ... and profile NCCL with nsys.
Challenge extension:
  - Re-run with per-GPU batch doubled and show efficiency recover toward 1.0 as
    compute grows relative to the (fixed) collective.

Run:  python IX.Production_Capstones/29.MultiGPU_Scaling/29.1_scaling_challenge.py
"""
print("=" * 70)
print("  Exercise 29.1 - Capstone 6: Multi-GPU Scaling Challenge")
print("=" * 70)


# ─────────────────────────────────────────────────────────────
# SECTION 1: Scaling efficiency from 1/2/4/8-GPU throughput
# ─────────────────────────────────────────────────────────────
print("\n-- Section 1: Scaling Efficiency --")
print("""
  Run a fixed per-GPU workload at 1/2/4/8 GPUs. Efficiency = speedup / N.
  Ideal is a straight line (eff 1.0); the gap is communication overhead.
""")

throughput = {1: 100, 2: 191, 4: 364, 8: 656}  # samples/s

def efficiency(tp):
    base = tp[1]
    return {n: round((v / base) / n, 3) for n, v in tp.items()}

eff = efficiency(throughput)
print(f"  {'GPUs':>5} {'tok/s':>8} {'speedup':>9} {'efficiency':>11}")
for n, v in throughput.items():
    print(f"  {n:>5} {v:>8} {v/throughput[1]:>8.2f}x {eff[n]:>10.2f}")
assert eff[1] == 1.0 and eff[8] >= 0.80
print("  [check] 8-GPU efficiency >= 0.80")


# ─────────────────────────────────────────────────────────────
# SECTION 2: NCCL AllReduce cost - NVLink vs InfiniBand
# ─────────────────────────────────────────────────────────────
print("\n-- Section 2: NCCL Communication Cost --")
print("""
  The scaling curve usually bends at the node boundary because intra-node NVLink
  is ~10x faster than inter-node InfiniBand for the same collective.
""")

def allreduce_ms(bytes_, n, bw_gbytes_s):
    moved = 2 * (n - 1) / n * bytes_
    return moved / (bw_gbytes_s * 1e9) * 1000

grad_bytes = 1_300_000_000   # ~1.3 GB gradient buffer
nv = allreduce_ms(grad_bytes, 8, 600)
ib = allreduce_ms(grad_bytes, 8, 50)
print(f"  AllReduce 1.3GB across 8 GPUs:")
print(f"    NVLink ~600 GB/s : {nv:7.2f} ms")
print(f"    InfiniBand ~50   : {ib:7.2f} ms  ({ib/nv:.0f}x slower)")
assert ib > nv * 5
print("  [check] inter-node collective is the likely 4->8 GPU efficiency drop")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Amdahl ceiling
# ─────────────────────────────────────────────────────────────
print("\n-- Section 3: Amdahl Speedup Ceiling --")
print("""
  Non-overlapped communication is a serial fraction s. Amdahl caps speedup at
  1 / (s + (1-s)/N), no matter how many GPUs you add.
""")

def amdahl(s, n):
    return 1.0 / (s + (1 - s) / n)

for s in (0.02, 0.05, 0.10):
    print(f"  serial fraction {s:.0%}: 8-GPU ceiling = {amdahl(s, 8):.2f}x, "
          f"infinite-GPU ceiling = {1/s:.1f}x")
assert amdahl(0.05, 8) < 8 and amdahl(0.02, 8) > amdahl(0.10, 8)
print("  [check] smaller serial fraction => higher achievable speedup")


print("\n" + "=" * 70)
print("  ALL SECTIONS PASSED - Exercise 29.1 complete")
print("  You computed scaling efficiency, attributed the gap to NCCL cost, and")
print("  bounded the speedup with Amdahl's law.")
print("=" * 70)
