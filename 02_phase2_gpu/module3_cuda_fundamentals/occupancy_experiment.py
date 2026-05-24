#!/usr/bin/env python3
"""
occupancy_experiment.py  ─  Phase 2 / Module 3: GPU Occupancy Deep Dive
=========================================================================

HOW TO RUN
    python occupancy_experiment.py
    python occupancy_experiment.py --exp batch_size   # occupancy via batch
    python occupancy_experiment.py --exp wave_count   # wave quantisation


"""

import argparse, math
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--exp", default="all",
    choices=["all","batch_size","wave_count","memory_pressure","occupancy_formula"])
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def sep(t): print(f"\n{'═'*60}\n  {t}\n{'─'*60}")
def cuda_ms(fn, w=5, n=30):
    for _ in range(w): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n): fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n

# ─────────────────────────────────────────────────────────────────────────────
# EXP 1 ─ Batch size → thread count → occupancy
# ─────────────────────────────────────────────────────────────────────────────
def exp_batch_size():
    sep("EXP 1 ─ Batch Size → Thread Count → Occupancy")
    print("""
  Every element in a GPU tensor = one thread (roughly).
  With a tiny batch (e.g. batch=1, seq=16) for LLM decode:
    Total threads = 1 × 16 × 4096 = 65,536
    RTX 4060 has 24 SMs × 1536 max threads/SM = 36,864 max threads
    → Only ~65K/36K = 1.8 waves → not all SMs stay busy

  With batch=32: 32 × 16 × 4096 = 2M threads → many waves → high occupancy

  We demonstrate this with a linear layer (matmul) at different batch sizes.
    """)
    if DEVICE != "cuda":
        print("  (Requires CUDA)"); return

    d = 4096      # LLM-like hidden dimension
    W = torch.randn(d, d, device=DEVICE, dtype=torch.float16)
    props = torch.cuda.get_device_properties(0)

    print(f"  GPU SMs: {props.multi_processor_count}  "
          f"Max threads/SM: {props.max_threads_per_multi_processor}")
    max_threads = props.multi_processor_count * props.max_threads_per_multi_processor
    print(f"  Max concurrent threads: {max_threads:,}\n")

    print(f"  {'Batch':>8}  {'Threads(est)':>14}  {'Waves':>8}  "
          f"{'Time(ms)':>10}  {'TFLOP/s':>10}  {'Util%':>8}")
    print(f"  {'─'*8}  {'─'*14}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*8}")

    peak_tflops = None
    for batch in [1, 2, 4, 8, 16, 32, 64, 128]:
        X = torch.randn(batch, d, device=DEVICE, dtype=torch.float16)
        ms  = cuda_ms(lambda: X @ W)
        flops = 2 * batch * d * d
        tflops = flops / (ms / 1000) / 1e12
        if peak_tflops is None or tflops > peak_tflops:
            peak_tflops = tflops
        threads = batch * d   # rough: one thread per output element
        waves   = threads / max_threads
        util    = tflops / peak_tflops * 100
        print(f"  {batch:>8}  {threads:>14,}  {waves:>8.2f}  "
              f"{ms:>10.3f}  {tflops:>10.2f}  {util:>7.0f}%")
        del X

    print(f"\n  → batch=1 decode step is severely under-occupying the GPU.")
    print(f"  → This is why continuous batching (vLLM) increases throughput.")

# ─────────────────────────────────────────────────────────────────────────────
# EXP 2 ─ Wave Quantisation
# ─────────────────────────────────────────────────────────────────────────────
def exp_wave_count():
    sep("EXP 2 ─ Wave Quantisation Effect")
    print("""
  The GPU assigns thread blocks to SMs in "waves".
  Each wave fills all SMs simultaneously.
  If total_blocks = n_SMs × k + remainder, the last partial wave
  still takes as long as a full wave → step function in execution time.

  Example (24 SMs):
    24 blocks → 1 full wave   → fast
    25 blocks → 1 wave + 1 SM → almost 2× slower for 1 extra block
    48 blocks → 2 full waves  → fast again

  This explains mysterious latency spikes when you change batch size by 1.
    """)
    if DEVICE != "cuda":
        print("  (Requires CUDA)"); return

    props = torch.cuda.get_device_properties(0)
    n_sms = props.multi_processor_count
    print(f"  GPU has {n_sms} SMs\n")
    print(f"  {'Blocks':>8}  {'Waves':>8}  {'Time(µs)':>10}  {'Note':>25}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*25}")

    # Use matrix row count as a proxy for "number of blocks"
    # (one block typically handles one or a few rows)
    for extra in range(0, n_sms + 5, max(1, n_sms//8)):
        n_blocks = n_sms + extra
        N = n_blocks * 32        # 32 elements per simulated block
        a = torch.rand(N, device=DEVICE)
        b = torch.rand(N, device=DEVICE)
        ms = cuda_ms(lambda: a.add_(b)) * 1000
        waves = n_blocks / n_sms
        note = "← partial wave" if (n_blocks % n_sms) != 0 else "full wave(s)"
        print(f"  {n_blocks:>8}  {waves:>8.2f}  {ms:>10.1f}  {note:>25}")
        del a, b

    print(f"\n  → Choose batch size to land on a full-wave multiple of {n_sms}.")

# ─────────────────────────────────────────────────────────────────────────────
# EXP 3 ─ Memory pressure limits occupancy
# ─────────────────────────────────────────────────────────────────────────────
def exp_memory_pressure():
    sep("EXP 3 ─ Memory Pressure and Occupancy")
    print("""
  When each thread/block allocates a lot of shared memory or registers,
  the SM can fit fewer simultaneous blocks → lower occupancy.

  For attention kernels:
    FlashAttention loads tiles into shared memory intentionally.
    The tile size controls the shared-memory usage per block.
    Larger tiles = better reuse, but fewer blocks per SM.
    Optimal tile size balances reuse vs occupancy.

  We simulate by varying the size of intermediate tensors (proxy for
  shared memory / register pressure in a custom kernel).
    """)
    if DEVICE != "cuda":
        print("  (Requires CUDA)"); return

    B, H, T = 8, 16, 64        # batch, heads, seq_len

    print(f"  {'head_dim':>10}  {'Time(ms)':>10}  {'FLOP/s(G)':>12}  {'Note':>30}")
    print(f"  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*30}")

    for D in [32, 64, 128, 256]:
        Q = torch.randn(B, H, T, D, device=DEVICE, dtype=torch.float16)
        K = torch.randn(B, H, T, D, device=DEVICE, dtype=torch.float16)
        V = torch.randn(B, H, T, D, device=DEVICE, dtype=torch.float16)

        scale = D ** -0.5
        def attn():
            s = torch.matmul(Q, K.transpose(-2,-1)) * scale
            s = torch.softmax(s, dim=-1)
            return torch.matmul(s, V)

        ms = cuda_ms(attn)
        # FLOPs: 2 matmuls each B*H*T*T*D
        flops = 2 * 2 * B * H * T * T * D
        gflops = flops / (ms/1000) / 1e9
        note = "more regs/block → fewer blocks" if D >= 128 else ""
        print(f"  {D:>10}  {ms:>10.3f}  {gflops:>12.1f}  {note:>30}")
        del Q, K, V

    print("""
  → Measure with ncu:
    ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active \\
        python occupancy_experiment.py --exp memory_pressure
    """)

# ─────────────────────────────────────────────────────────────────────────────
# EXP 4 ─ Occupancy formula (theoretical calculation)
# ─────────────────────────────────────────────────────────────────────────────
def exp_occupancy_formula():
    sep("EXP 4 ─ Theoretical Occupancy Formula")
    print("""
  You can predict occupancy before running:

  For each SM:
    max_warps       = max_threads_per_SM / 32
    warps_per_block = block_size / 32
    blocks_per_SM   = min(
        max_blocks_per_SM,                    # hardware limit (~32)
        floor(max_warps / warps_per_block),   # warp limit
        floor(shared_mem_per_SM / smem_per_block),  # shared mem limit
        floor(max_regs_per_SM  / regs_per_block),   # register limit
    )
    occupancy = blocks_per_SM × warps_per_block / max_warps
    """)

    if DEVICE != "cuda":
        print("  (Requires CUDA)"); return

    p = torch.cuda.get_device_properties(0)
    max_warps  = p.max_threads_per_multi_processor // 32
    max_blocks = 32   # typical Ampere/Ada SM limit

    print(f"  GPU: {p.name}")
    print(f"  max_threads/SM = {p.max_threads_per_multi_processor}")
    print(f"  max_warps/SM   = {max_warps}")
    print(f"  max_blocks/SM  = {max_blocks} (approximate)\n")

    print(f"  {'block_size':>12}  {'warps/blk':>10}  {'blks/SM':>8}  "
          f"{'occ%':>6}  {'recommendation':>20}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*8}  {'─'*6}  {'─'*20}")

    for bs in [32, 64, 128, 256, 512, 1024]:
        warps_blk = bs // 32
        blks_sm   = min(max_blocks, max_warps // warps_blk)
        occ       = (blks_sm * warps_blk / max_warps) * 100
        rec = "too small" if bs < 64 else ("good" if occ >= 75 else "check registers")
        print(f"  {bs:>12}  {warps_blk:>10}  {blks_sm:>8}  {occ:>5.0f}%  {rec:>20}")

    print("""
  General rule for simple memory-bound kernels (like vector add):
    block_size = 256 gives ~100% occupancy on most modern GPUs.
  For compute-heavy kernels (attention):
    Use ncu's occupancy limiter analysis to find the true bottleneck.
    """)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  occupancy_experiment.py  ─  Phase 2 Module 3")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}")

    dispatch = {
        "batch_size":         exp_batch_size,
        "wave_count":         exp_wave_count,
        "memory_pressure":    exp_memory_pressure,
        "occupancy_formula":  exp_occupancy_formula,
    }
    if args.exp == "all":
        for fn in dispatch.values(): fn()
    else:
        dispatch[args.exp]()
