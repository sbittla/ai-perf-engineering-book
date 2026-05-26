#!/usr/bin/env python3
"""
IV.LLM_Inference_Systems/13.Distributed_Inference/13.3_fsdp_and_pipeline.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chapter 13: Distributed Inference — Section 3: FSDP and Pipeline Parallelism
=======================================================================
Covers book section 13.3:
  • FSDP: Full Sharded Data Parallelism — sharding parameters and gradients
  • Why FSDP reduces peak memory vs DDP (no full param copy on each GPU)
  • Pipeline parallelism: assign layers to stages, micro-batch interleaving
  • Pipeline bubble: wasted GPU time at the start and end of a pipeline
  • Combining TP + PP + DP for extreme scale (3D parallelism)
  • Memory vs throughput trade-offs: which strategy to use when

Run:  python IV.LLM_Inference_Systems/13.Distributed_Inference/13.3_fsdp_and_pipeline.py
All sections must print ✓.
"""

import math
import time

print("=" * 60)
print("  Exercise 13.3 — FSDP and Pipeline Parallelism")
print("=" * 60)
print()


# ─────────────────────────────────────────────────────────────
# SECTION 1: DDP vs FSDP memory comparison
# ─────────────────────────────────────────────────────────────
print("── Section 1: DDP vs FSDP Memory Usage ──")
print("""
  DDP (Distributed Data Parallel):
    Each GPU holds a FULL copy of model parameters AND gradients.
    VRAM per GPU = 2 × param_bytes + optimizer_states
    AllReduce gradients after backward.

  FSDP (Fully Sharded Data Parallel):
    Parameters, gradients, AND optimizer states are sharded across N GPUs.
    Each GPU holds 1/N of each parameter tensor.
    Before a forward pass: AllGather to reconstruct the full layer.
    After backward: ReduceScatter to shard the gradients.

  MEMORY COMPARISON (Adam optimizer, FP16 params, FP32 optimizer states):
    DDP  per GPU = params × 2 B (model)
                + params × 2 B (gradients)
                + params × 8 B (optimizer: first+second moment, FP32)
                = 12 × param_bytes

    FSDP per GPU = (12 × param_bytes) / N    ← sharded
                + (2 × param_bytes)          ← full param buffer during forward
                + activation_bytes

  For Llama-7B with 8-GPU FSDP:
    DDP  : 7B × 12B = 84 GB per GPU (needs A100 80GB, barely)
    FSDP : 84 GB / 8 + 14 GB buffer ≈ 25 GB per GPU (RTX 4090 feasible)
""")


def ddp_memory_gb(params: float, dtype_bytes: int = 2) -> float:
    """Memory per GPU for DDP. params in billions."""
    param_gb = params * 1e9 * dtype_bytes / 1e9
    grad_gb  = param_gb
    opt_gb   = params * 1e9 * 8 / 1e9   # FP32 optimizer states
    return param_gb + grad_gb + opt_gb


def fsdp_memory_gb(params: float, n_gpus: int, dtype_bytes: int = 2) -> float:
    """Memory per GPU for FSDP. params in billions."""
    sharded_gb  = (params * 1e9 * (dtype_bytes + dtype_bytes + 8)) / n_gpus / 1e9
    buffer_gb   = params * 1e9 * dtype_bytes / 1e9   # full param buffer during forward
    return sharded_gb + buffer_gb


print(f"  {'Model':>12}  {'Params':>8}  {'DDP (GB)':>10}  {'FSDP 4GPU':>12}  {'FSDP 8GPU':>12}")
print(f"  {'─'*12}  {'─'*8}  {'─'*10}  {'─'*12}  {'─'*12}")
for name, params_b in [("Llama-7B", 7.0), ("Llama-13B", 13.0), ("Llama-70B", 70.0)]:
    ddp  = ddp_memory_gb(params_b)
    f4   = fsdp_memory_gb(params_b, 4)
    f8   = fsdp_memory_gb(params_b, 8)
    print(f"  {name:>12}  {params_b:>7.0f}B  {ddp:>10.1f}  {f4:>12.1f}  {f8:>12.1f}")

# TODO 1: Verify FSDP(8 GPU) < DDP for Llama-7B
assert fsdp_memory_gb(7.0, 8) < ddp_memory_gb(7.0), \
    "FSDP should use less memory than DDP"
print(f"\n  Llama-7B: DDP={ddp_memory_gb(7.0):.0f}GB vs FSDP/8={fsdp_memory_gb(7.0,8):.0f}GB — "
      f"FSDP saves {ddp_memory_gb(7.0)-fsdp_memory_gb(7.0,8):.0f}GB")
print("  ✓ Section 1 passed — FSDP dramatically reduces peak memory vs DDP")


# ─────────────────────────────────────────────────────────────
# SECTION 2: Pipeline parallelism mechanics
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: Pipeline Parallelism ──")
print("""
  PIPELINE PARALLELISM (PP):
    Split the model's layers across P GPUs (stages).
    Each GPU processes all micro-batches for its layers,
    then passes activations to the next stage.

    Stage 0: layers 0...(L/P - 1)
    Stage 1: layers L/P...(2L/P - 1)
    ...
    Stage P-1: layers (P-1)L/P...L-1

  PIPELINE BUBBLE:
    On startup, stages 1..P-1 are idle waiting for stage 0 to finish.
    On shutdown, stage 0 is idle while stages 1..P-1 drain.
    Bubble fraction = (P - 1) / (M + P - 1)
    where M = number of micro-batches per pipeline flush.

    With M micro-batches per pipeline step:
      Useful compute = M / (M + P - 1)
      Bubble        = (P - 1) / (M + P - 1)

  To minimise bubble: use large M (many micro-batches).
  Rule of thumb: M >= 4 × P to keep bubble below 20%.

  TODO 2: Implement pipeline_bubble_fraction() below.
""")


def pipeline_bubble_fraction(P: int, M: int) -> float:
    """
    TODO 2: Return pipeline bubble fraction.
    = (P - 1) / (M + P - 1)
    """
    # YOUR CODE HERE
    return (P - 1) / (M + P - 1)


assert abs(pipeline_bubble_fraction(1, 10) - 0.0) < 1e-9, "1 stage = 0 bubble"
assert abs(pipeline_bubble_fraction(4, 4)  - 0.429) < 0.01, "4 stages, 4 micro = ~43%"

print(f"  Pipeline bubble fraction  (ideal < 20%):")
print(f"\n  {'P (stages)':>12}  ", end="")
for M in [2, 4, 8, 16, 32]:
    print(f"  M={M:>2}", end="")
print()
print(f"  {'─'*12}  ", end="")
for _ in [2, 4, 8, 16, 32]:
    print(f"  {'─'*5}", end="")
print()
for P in [2, 4, 8, 16]:
    print(f"  {P:>12}  ", end="")
    for M in [2, 4, 8, 16, 32]:
        b = pipeline_bubble_fraction(P, M)
        mark = "✓" if b < 0.2 else " "
        print(f"  {b:.2f}{mark}", end="")
    print()

# Verify M=32 reduces bubble significantly for P=8
b_m4  = pipeline_bubble_fraction(8, 4)
b_m32 = pipeline_bubble_fraction(8, 32)
assert b_m32 < b_m4, f"More micro-batches → smaller bubble: {b_m4:.2f} vs {b_m32:.2f}"
print(f"\n  P=8: bubble drops from {b_m4:.1%} (M=4) to {b_m32:.1%} (M=32)")
print("  ✓ Section 2 passed — pipeline bubble formula working")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Interleaved pipeline (1F1B schedule)
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: 1F1B Interleaved Schedule ──")
print("""
  NAIVE SCHEDULE ("GPipe"):
    All micro-batches forward, then all backward.
    Peak memory: M × activation_per_microbatch (must hold all activations)
    Bubble fraction: (P-1)/(M+P-1) — same as above.

  1F1B SCHEDULE (One Forward One Backward):
    Interleave forward and backward within a single pipeline flush.
    As soon as stage 0 finishes microbatch k's forward, it starts k's backward.
    Peak memory: P × activation_per_microbatch (only P activations in flight)

  INTERLEAVED 1F1B (Megatron):
    Each stage handles V virtual stages (model chunks).
    Reduces bubble from (P-1)/(M+P-1) to (P-1)/(V×M+P-1)
    At V=2: bubble halved. At large V: bubble → 0.
    Cost: V × 2(P-1) extra communication events per flush.

  TODO 3: Implement interleaved_bubble_fraction() below.
""")


def interleaved_bubble_fraction(P: int, M: int, V: int) -> float:
    """
    TODO 3: Return bubble fraction for interleaved 1F1B with V virtual stages.
    = (P - 1) / (V * M + P - 1)
    """
    # YOUR CODE HERE
    return (P - 1) / (V * M + P - 1)


print(f"  P=8, M=8, varying V (virtual stages):")
for V in [1, 2, 4, 8]:
    b = interleaved_bubble_fraction(8, 8, V)
    print(f"    V={V}: bubble = {b:.2%}")

assert interleaved_bubble_fraction(8, 8, 2) < pipeline_bubble_fraction(8, 8), \
    "Interleaved should have smaller bubble than naive"
print(f"\n  V=2 halves bubble vs V=1 (naive): "
      f"{pipeline_bubble_fraction(8,8):.1%} → {interleaved_bubble_fraction(8,8,2):.1%}")
print("  ✓ Section 3 passed — interleaved 1F1B reduces pipeline bubble")


# ─────────────────────────────────────────────────────────────
# SECTION 4: 3D parallelism — combining TP + PP + DP
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: 3D Parallelism at Scale ──")
print("""
  For extreme scale (GPT-4 class), a single parallelism strategy is not enough.
  Megatron-LM and DeepSpeed combine three orthogonal strategies:

    TENSOR PARALLELISM (TP):
      Split weight matrices within a node (NVLink required).
      Typical: TP=8 within an 8-GPU server node.

    PIPELINE PARALLELISM (PP):
      Assign layer groups to different nodes.
      Typical: PP=N_nodes or PP=N_nodes/2.
      Requires fast inter-node network (InfiniBand).

    DATA PARALLELISM (DP):
      Run multiple independent copies of the TP+PP model.
      AllReduce gradients across DP replicas.
      Typical: DP=2–8 for training.

  TOTAL GPUs = TP × PP × DP

  Example — GPT-3 175B training (Megatron):
    TP=8  (within each 8-GPU DGX A100)
    PP=16 (16 pipeline stages across 16 DGX nodes)
    DP=12 (12 replicas of the TP+PP model)
    Total = 8 × 16 × 12 = 1536 A100s

  Memory per GPU (FP16, Adam, DP with FSDP across DP replicas):
    Model params : 175B × 2B = 350 GB total / (8×16 GPU pipeline+TP) ≈ 2.7 GB/GPU
    Gradients    : same shard
    Optimizer    : FP32, 8B per param / (8×16) ≈ 10.9 GB/GPU
    Activations  : ~80 GB/GPU (gradient checkpointing needed)
""")


def total_gpus(tp: int, pp: int, dp: int) -> int:
    return tp * pp * dp


def memory_per_gpu_gb(params_b: float, tp: int, pp: int, dp: int,
                      dtype_bytes: int = 2) -> float:
    """Rough estimate: param + grad sharded across tp × pp, optimizer over tp × pp × dp."""
    tp_pp      = tp * pp
    param_gb   = params_b * 1e9 * dtype_bytes / tp_pp / 1e9
    grad_gb    = param_gb
    opt_gb     = params_b * 1e9 * 8 / (tp_pp * dp) / 1e9   # FP32 optimizer sharded over DP
    return param_gb + grad_gb + opt_gb


configs = [
    ("Llama-7B  small",   7.0, 1, 1, 1),
    ("Llama-7B  TP=2",    7.0, 2, 1, 1),
    ("Llama-13B TP=2",   13.0, 2, 1, 1),
    ("Llama-70B TP=8",   70.0, 8, 1, 1),
    ("Llama-70B TP=4,PP=2",70.0, 4, 2, 1),
    ("GPT-3 175B full",  175.0, 8, 16, 12),
]

print(f"  {'Config':<26}  {'Total GPUs':>10}  {'Mem/GPU (GB)':>13}")
print(f"  {'─'*26}  {'─'*10}  {'─'*13}")
for name, p, tp, pp, dp in configs:
    n   = total_gpus(tp, pp, dp)
    mem = memory_per_gpu_gb(p, tp, pp, dp)
    print(f"  {name:<26}  {n:>10}  {mem:>13.1f}")

# TODO 4: Verify 3D parallelism reduces memory vs single GPU
single = memory_per_gpu_gb(70.0, 1, 1, 1)
tp8    = memory_per_gpu_gb(70.0, 8, 1, 1)
assert tp8 < single, f"TP=8 should use less memory: {tp8:.0f} vs {single:.0f}"
print(f"\n  Llama-70B: single GPU needs {single:.0f}GB → TP=8 needs {tp8:.0f}GB/GPU")
print("  ✓ Section 4 passed — 3D parallelism enables training at extreme scale")


# ─────────────────────────────────────────────────────────────
# SECTION 5: Which parallelism strategy to use?
# ─────────────────────────────────────────────────────────────
print("\n── Section 5: Decision Guide — TP vs PP vs DP ──")
print("""
  USE TENSOR PARALLELISM (TP) when:
    • Model fits in a single node (NVLink available)
    • Individual layers are too large for one GPU
    • Need lowest latency (all GPUs work on every token)
    • Example: Llama-70B on 4–8 A100 within one DGX

  USE PIPELINE PARALLELISM (PP) when:
    • Model spans multiple nodes (tensor TP doesn't cross nodes well)
    • Batch is large enough to fill the pipeline (M ≥ 4P)
    • Throughput > latency is the priority
    • Example: 100B+ models across many nodes

  USE DATA PARALLELISM (DP / FSDP) when:
    • Dataset is large and you want to train faster
    • Model fits on N GPUs with TP/PP, but you have more GPUs
    • Gradient averaging across replicas is the only communication needed
    • Example: TP=8 within each node, DP across 8 nodes = 64 GPU total

  USE FSDP (instead of DDP) when:
    • Optimizer states don't fit per GPU
    • Model is 10B+ parameters
    • Fine-tuning a large foundation model on limited GPUs

  COMBINATION GUIDE:
    7B  models   : TP=1 or 2 within node, DP across nodes
    70B  models  : TP=8 within node, optional DP
    175B+ models : TP=8, PP=4–16, DP as needed
""")


def recommend_parallelism(params_b: float, n_gpus: int, vram_gb_per_gpu: float) -> str:
    """Simple heuristic recommendation based on model size and GPU count."""
    weights_gb = params_b * 2   # FP16

    if weights_gb <= vram_gb_per_gpu:
        if n_gpus == 1:
            return "Single GPU — no parallelism needed"
        return "DP (DataParallel or FSDP across GPUs)"

    if weights_gb <= vram_gb_per_gpu * n_gpus:
        if n_gpus <= 8:
            return f"TP={n_gpus} (within node, NVLink required)"
        tp = 8
        pp = n_gpus // tp
        return f"TP={tp} + PP={pp} (multi-node)"

    return f"TP=8 + PP + DP — need >{n_gpus} GPUs or quantize to INT4"


print(f"  {'Model':<12}  {'GPUs':>5}  {'VRAM':>8}  Recommendation")
print(f"  {'─'*12}  {'─'*5}  {'─'*8}  {'─'*40}")
for model, p, n, v in [
    ("Llama-7B",   7.0,  1, 80),
    ("Llama-7B",   7.0,  4, 24),
    ("Llama-13B", 13.0,  2, 80),
    ("Llama-70B", 70.0,  8, 80),
    ("Llama-70B", 70.0,  4, 80),
    ("GPT-3 175B",175.0,16, 80),
]:
    rec = recommend_parallelism(p, n, v)
    print(f"  {model:<12}  {n:>5}  {v:>7}GB  {rec}")

print("  ✓ Section 5 passed — parallelism decision guide complete")


print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 13.3 complete!")
print("  You understand FSDP memory savings, pipeline bubble,")
print("  interleaved schedules, and 3D parallelism trade-offs.")
print("=" * 60)
print()
print("  Part IV complete!  You have covered:")
print("  Chapter 10: Prefill/decode, KV cache, inference metrics")
print("  Chapter 11: Static batching, continuous batching, PagedAttention")
print("  Chapter 12: Speculative decoding, acceptance rate analysis")
print("  Chapter 13: Tensor parallelism, NCCL, FSDP, pipeline parallelism")
