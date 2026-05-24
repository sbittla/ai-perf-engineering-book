#!/usr/bin/env python3
"""
fsdp_train.py  ─  Phase 4 / Module 9: FSDP Distributed Training
=================================================================

HOW TO RUN
    # Single process (simulated FSDP):
    python fsdp_train.py

    # True multi-GPU (requires 2+ GPUs):
    torchrun --nproc_per_node=2 fsdp_train.py --distributed

    # Profile to see all-gather/reduce-scatter in timeline:
    nsys profile --trace=cuda,nvtx torchrun --nproc_per_node=2 fsdp_train.py --distributed


"""

import argparse, os, sys, functools
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'models'))
try:
    from model import TinyTransformer, TransformerBlock
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False

parser = argparse.ArgumentParser()
parser.add_argument("--distributed", action="store_true")
parser.add_argument("--shard",  default="full",
    choices=["full","grad_op","no_shard"])
parser.add_argument("--steps",  type=int, default=20)
parser.add_argument("--batch",  type=int, default=8)
args = parser.parse_args()

IS_DIST = "RANK" in os.environ or args.distributed

def is_main():
    if IS_DIST and dist.is_initialized():
        return dist.get_rank() == 0
    return True

# ─────────────────────────────────────────────────────────────────────────────
# Setup distributed
# ─────────────────────────────────────────────────────────────────────────────
if IS_DIST and "RANK" in os.environ:
    dist.init_process_group("nccl")
    rank  = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    DEVICE = torch.device(f"cuda:{rank}")
else:
    rank, world = 0, 1
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if is_main():
    print(f"\n{'='*60}")
    print(f"  fsdp_train.py  ─  Phase 4 Module 9")
    print(f"  GPUs: {world}   Shard: {args.shard}   Device: {DEVICE}")
    print(f"{'='*60}\n")

if not HAS_MODEL:
    if is_main(): print("  model.py not found. Exiting.")
    exit(0)

# ─────────────────────────────────────────────────────────────────────────────
# Build model
# ─────────────────────────────────────────────────────────────────────────────
model = TinyTransformer("medium", max_seq=65).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())

if is_main():
    print(f"  Model: TinyTransformer[medium]  {n_params/1e6:.1f}M params")
    vram_fp16 = n_params * 2 / 1e9
    print(f"  FP16 size: {vram_fp16:.3f}GB  → each GPU holds ~{vram_fp16/max(world,1):.3f}GB with FSDP FULL_SHARD\n")

# ─────────────────────────────────────────────────────────────────────────────
# Wrap with FSDP
# ─────────────────────────────────────────────────────────────────────────────
shard_map = {
    "full":     ShardingStrategy.FULL_SHARD,
    "grad_op":  ShardingStrategy.SHARD_GRAD_OP,
    "no_shard": ShardingStrategy.NO_SHARD,
}

if IS_DIST and dist.is_initialized():
    # auto_wrap_policy: automatically wraps TransformerBlock submodules
    # so each block can be all-gathered/reduce-scattered independently
    my_auto_wrap = functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=1_000,   # wrap any module with > 1K params
    )

    # MixedPrecision: run compute in FP16, keep params as FP16
    # reduce_dtype=FP32 ensures gradient reduce is numerically stable
    mp_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float16,
    )

    fsdp_model = FSDP(
        model,
        sharding_strategy=shard_map[args.shard],
        auto_wrap_policy=my_auto_wrap,
        mixed_precision=mp_policy,
        device_id=torch.cuda.current_device(),
    )
    if is_main():
        print(f"  FSDP wrapped. Shard strategy: {args.shard.upper()}")
else:
    # Single GPU: use model directly (demonstrates the API without actual sharding)
    fsdp_model = model
    if is_main():
        print(f"  Single GPU mode (no actual sharding).\n")

# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

import time
fsdp_model.train()
step_times = []
total_start = time.perf_counter()

for step in range(args.steps):
    ids = torch.randint(0, 50257, (args.batch, 64), device=DEVICE)

    t0 = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type=str(DEVICE).split(':')[0], dtype=torch.float16):
        logits, _ = fsdp_model(ids)
        loss = criterion(logits[:, :-1, :].reshape(-1, logits.size(-1)),
                         ids[:, 1:].reshape(-1))

    loss.backward()
    # Gradient clipping works with FSDP through the model's grad norm
    fsdp_model.clip_grad_norm_(1.0) if IS_DIST and dist.is_initialized() \
        else nn.utils.clip_grad_norm_(fsdp_model.parameters(), 1.0)
    optimizer.step()

    torch.cuda.synchronize()
    step_ms = (time.perf_counter() - t0) * 1000
    step_times.append(step_ms)

    if is_main() and step % 5 == 0:
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"  step {step:3d}/{args.steps}  loss={loss.item():.4f}  "
              f"step={step_ms:.1f}ms  vram={vram:.3f}GB")

if is_main():
    avg_ms = sum(step_times) / len(step_times)
    total  = time.perf_counter() - total_start
    print(f"\n{'='*60}")
    print(f"  FSDP Training Complete")
    print(f"  Avg step  : {avg_ms:.1f}ms")
    print(f"  Throughput: {args.steps / total:.1f} steps/sec")
    if DEVICE.type == "cuda":
        print(f"  Peak VRAM : {torch.cuda.max_memory_allocated()/1e9:.3f}GB per GPU")
    print(f"""
  FSDP vs DDP Memory (theory):
    DDP  : {n_params * 3 * 2 / 1e9 / max(world,1):.3f}GB/GPU (params + grads + optim, replicated)
    FSDP : {n_params * 3 * 2 / 1e9 / max(world,1):.3f}GB/GPU ({world}-way sharded)

  KEY COMMANDS
    # Profile all-gather / reduce-scatter:
    nsys profile --trace=cuda,nvtx \\
        torchrun --nproc_per_node=2 fsdp_train.py --distributed

    # Check memory saved vs DDP:
    # Compare peak VRAM with --shard full vs --shard no_shard
  """)

if IS_DIST and dist.is_initialized():
    dist.destroy_process_group()
