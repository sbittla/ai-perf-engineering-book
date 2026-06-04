#!/usr/bin/env python3
"""
26.Distributed_Training/26.1_scaling_analysis.py  —  Chapter 26: Distributed Training
=======================================================================
Covers book sections 26.1 - 26.5:
  26.1  DDP (replicate + AllReduce, overlapped with backward)
  26.2  Sharding: FSDP and ZeRO stages 1/2/3
  26.3  The cost of AllReduce (~2S/B)
  26.4  Gradient accumulation and comm/compute overlap
  26.5  Measuring scaling efficiency

Difficulty: ***--  (3/5 - analytic, no multi-GPU needed)
Est. time:  40-50 minutes
Expected ranges:
  - Healthy scaling efficiency at 8 GPUs: >= 0.80
  - ZeRO-3 per-GPU param memory: ~1/N of the full model
Troubleshooting:
  - No multi-GPU hardware needed; everything is analytic.
  - If you adapt this to real torchrun, set NCCL_DEBUG=INFO to see collectives.
Challenge extension:
  - Add a comm/compute-overlap factor and show how raising per-GPU batch size
    moves efficiency from 0.6 back above 0.8.

Run:  python VIII.Advanced_Performance/26.Distributed_Training/26.1_scaling_analysis.py
"""
print("=" * 70)
print("  Exercise 26.1 - Distributed Training: Scaling Analysis")
print("=" * 70)


# ─────────────────────────────────────────────────────────────
# SECTION 1: AllReduce cost on NVLink vs InfiniBand
# ─────────────────────────────────────────────────────────────
print("\n-- Section 1: Ring AllReduce Cost --")
print("""
  Ring AllReduce of S bytes across N GPUs moves ~2(N-1)/N * S bytes per GPU,
  so time ~= 2(N-1)/N * S / B for interconnect bandwidth B.
""")

def allreduce_ms(param_count, dtype_bytes, n_gpus, bw_gbytes_s):
    S = param_count * dtype_bytes
    bytes_moved = 2 * (n_gpus - 1) / n_gpus * S
    return bytes_moved / (bw_gbytes_s * 1e9) * 1000

P = 7_000_000_000     # 7B params
for name, bw in [("NVLink ~600 GB/s", 600), ("InfiniBand ~50 GB/s", 50)]:
    ms = allreduce_ms(P, 2, 8, bw)     # bf16 grads, 8 GPUs
    print(f"  {name:<22} 7B bf16 grads, 8 GPUs:  {ms:8.1f} ms / step")
nv = allreduce_ms(P, 2, 8, 600); ib = allreduce_ms(P, 2, 8, 50)
assert ib > nv * 5, "InfiniBand AllReduce should be much slower than NVLink"
print("  [check] AllReduce cost scales inversely with interconnect bandwidth")


# ─────────────────────────────────────────────────────────────
# SECTION 2: ZeRO per-GPU memory by stage
# ─────────────────────────────────────────────────────────────
print("\n-- Section 2: ZeRO Per-GPU Memory --")
print("""
  Adam training state per parameter (mixed precision, rough): ~2 bytes weights
  + ~2 grads + ~12 optimizer (fp32 master + m + v) = ~16 bytes/param.
  DDP replicates all of it; ZeRO shards progressively across N GPUs.
""")

def per_gpu_bytes(P, n, stage):
    w, g, opt = 2, 2, 12   # bytes/param
    if stage == "ddp":     return P * (w + g + opt)
    if stage == "zero1":   return P * (w + g + opt / n)
    if stage == "zero2":   return P * (w + g / n + opt / n)
    if stage == "zero3":   return P * (w / n + g / n + opt / n)
    raise ValueError(stage)

n = 8
for s in ("ddp", "zero1", "zero2", "zero3"):
    gb = per_gpu_bytes(P, n, s) / 1e9
    print(f"  {s:<6} (7B, {n} GPUs): {gb:7.1f} GB / GPU")
assert per_gpu_bytes(P, n, "zero3") < per_gpu_bytes(P, n, "ddp")
assert per_gpu_bytes(P, n, "zero3") < per_gpu_bytes(P, n, "zero1")
print("  [check] memory falls monotonically DDP -> ZeRO-1 -> 2 -> 3")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Scaling efficiency + Amdahl ceiling
# ─────────────────────────────────────────────────────────────
print("\n-- Section 3: Scaling Efficiency --")
print("""
  Efficiency = (speedup on N GPUs) / N. 1.0 is perfect; keep it >= 0.8.
""")

def scaling_efficiency(tp_by_gpus):
    base = tp_by_gpus[1]
    return {n: round((tp / base) / n, 3) for n, tp in tp_by_gpus.items()}

def amdahl_max_speedup(serial_frac, n):
    return 1.0 / (serial_frac + (1 - serial_frac) / n)

tp = {1: 100, 2: 190, 4: 360, 8: 640}
eff = scaling_efficiency(tp)
print(f"  measured throughput: {tp}")
print(f"  efficiency:          {eff}")
print(f"  Amdahl ceiling @5% serial, 8 GPUs: {amdahl_max_speedup(0.05, 8):.2f}x")
assert eff[1] == 1.0 and eff[8] >= 0.80
assert amdahl_max_speedup(0.05, 8) < 8
print("  [check] efficiency computed; 8-GPU efficiency >= 0.80")


print("\n" + "=" * 70)
print("  ALL SECTIONS PASSED - Exercise 26.1 complete")
print("  You modeled AllReduce cost, ZeRO memory by stage, and scaling")
print("  efficiency with an Amdahl ceiling.")
print("=" * 70)
