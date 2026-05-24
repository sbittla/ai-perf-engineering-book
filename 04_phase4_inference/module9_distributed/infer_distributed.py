#!/usr/bin/env python3
"""
infer_distributed.py  —  Phase 4, Module 9: Multi-GPU Tensor Parallel Inference
==================================================================================

HOW TO RUN:
    # Single GPU (baseline):
    python infer_distributed.py --gpus 1

    # Multi-GPU with torchrun (requires N physical GPUs):
    torchrun --nproc_per_node=2 infer_distributed.py

    # With NCCL debug logging:
    NCCL_DEBUG=INFO torchrun --nproc_per_node=2 infer_distributed.py

    # Profile NCCL communication:
    NCCL_DEBUG=INFO nsys profile --trace=cuda,nvtx,nccl \\
        torchrun --nproc_per_node=2 infer_distributed.py


"""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'models'))
from model import TinyTransformer

parser = argparse.ArgumentParser()
parser.add_argument("--model",   default="small", choices=["tiny","small","medium","large"])
parser.add_argument("--tokens",  type=int, default=50)
parser.add_argument("--batch",   type=int, default=4)
parser.add_argument("--runs",    type=int, default=5)
args = parser.parse_args()

# ── Detect if we're in a torchrun context ─────────────────────────────────────
IS_DISTRIBUTED = "RANK" in os.environ

def is_main():
    """True if this is rank 0 (or non-distributed mode)."""
    if IS_DISTRIBUTED:
        return dist.get_rank() == 0
    return True


# =============================================================================
# Tensor-Parallel Linear Layer
# =============================================================================

class ColumnParallelLinear(nn.Module):
    """
    Column-parallel linear: splits the output dimension across GPUs.
    Each GPU computes a subset of the output features.
    No all-reduce needed here — outputs are gathered by RowParallelLinear.

    Example: W shape (d_model=512, d_out=512), 2 GPUs:
        GPU 0: W_local = W[:, :256]  → output shape (B, T, 256)
        GPU 1: W_local = W[:, 256:]  → output shape (B, T, 256)
    """
    def __init__(self, in_features, out_features, world_size, rank):
        super().__init__()
        assert out_features % world_size == 0, "out_features must be divisible by world_size"
        self.local_out = out_features // world_size
        self.weight = nn.Parameter(torch.randn(in_features, self.local_out) * 0.02)
        self.bias   = nn.Parameter(torch.zeros(self.local_out))

    def forward(self, x):
        return x @ self.weight + self.bias


class RowParallelLinear(nn.Module):
    """
    Row-parallel linear: splits the input dimension across GPUs.
    Each GPU has a slice of the input → computes partial output → all-reduce sums.

    Example: W shape (d_in=512, d_out=512), 2 GPUs:
        GPU 0: W_local = W[:256, :]  → partial = x_local @ W_local
        GPU 1: W_local = W[256:, :]  → partial = x_local @ W_local
        All-reduce: y = sum(partials across all GPUs)
    """
    def __init__(self, in_features, out_features, world_size, rank):
        super().__init__()
        assert in_features % world_size == 0
        self.local_in = in_features // world_size
        self.weight   = nn.Parameter(torch.randn(self.local_in, out_features) * 0.02)

    def forward(self, x_local):
        # x_local: (B, T, local_in) — this GPU's slice of the input
        partial = x_local @ self.weight   # (B, T, out_features)

        if IS_DISTRIBUTED:
            # ALL-REDUCE: sum partial results across all GPUs
            # After this, every GPU has the complete output
            # This is the communication cost of tensor parallelism
            dist.all_reduce(partial, op=dist.ReduceOp.SUM)

        return partial


# =============================================================================
# Tensor-Parallel Attention (simplified)
# =============================================================================

class TensorParallelAttention(nn.Module):
    """
    Multi-head attention with heads split across GPUs.

    With n_heads=8 and 2 GPUs:
        GPU 0: handles heads 0, 1, 2, 3
        GPU 1: handles heads 4, 5, 6, 7

    Each GPU:
        1. Projects Q, K, V for its heads (ColumnParallelLinear)
        2. Computes attention scores and weighted values
        3. Projects output back to full d_model (RowParallelLinear + all-reduce)
    """
    def __init__(self, d_model, n_heads, world_size, rank):
        super().__init__()
        assert n_heads % world_size == 0
        self.d_model   = d_model
        self.n_heads   = n_heads
        self.local_heads = n_heads // world_size
        self.head_dim  = d_model // n_heads
        self.world_size = world_size

        # Q, K, V projections — column parallel (split output dim)
        local_d = self.local_heads * self.head_dim
        self.q_proj = nn.Linear(d_model, local_d, bias=False)
        self.k_proj = nn.Linear(d_model, local_d, bias=False)
        self.v_proj = nn.Linear(d_model, local_d, bias=False)

        # Output projection — row parallel (all-reduce on output)
        self.out_proj = RowParallelLinear(local_d, d_model, world_size, rank)

    def forward(self, x):
        B, T, _ = x.shape
        H = self.local_heads
        D = self.head_dim

        # Each GPU computes Q, K, V for its own heads
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        # Local attention computation (no communication needed here)
        scale = D ** -0.5
        attn  = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        y     = (attn @ v).transpose(1, 2).contiguous().view(B, T, H * D)

        # Output projection includes all-reduce (RowParallelLinear)
        return self.out_proj(y)


# =============================================================================
# Benchmark: Single GPU vs Distributed
# =============================================================================

def run_single_gpu(model_size, batch, max_new_tokens, n_runs):
    """Baseline: standard single-GPU inference."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model  = TinyTransformer(size=model_size, max_seq=256).to(device)
    model.eval()

    prompt = torch.randint(0, 50257, (batch, 32), device=device)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            model.generate(prompt, max_new_tokens=max_new_tokens, temperature=0)

    times = []
    for _ in range(n_runs):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        with torch.no_grad():
            out = model.generate(prompt, max_new_tokens=max_new_tokens, temperature=0)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))

    n_new = out.shape[1] - prompt.shape[1]
    return {
        "avg_ms":  sum(times) / len(times),
        "tps":     n_new / (sum(times) / len(times) / 1000),
        "n_tokens": n_new,
    }


def main():
    if IS_DISTRIBUTED:
        # ── Distributed mode (launched with torchrun) ─────────────────────────
        dist.init_process_group(backend="nccl")
        rank      = dist.get_rank()
        world     = dist.get_world_size()
        device    = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        rank, world = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_main():
        print(f"\n{'='*60}")
        print(f"  Multi-GPU Distributed Inference")
        print(f"{'='*60}")
        print(f"  GPUs (world size) : {world}")
        print(f"  Model             : {args.model}")
        print(f"  Batch             : {args.batch}")
        print(f"  Tokens to generate: {args.tokens}")
        print(f"{'='*60}\n")

    # Build a simple model with tensor-parallel attention for the demo
    d_model = {"tiny": 128, "small": 256, "medium": 512, "large": 768}[args.model]
    n_heads = {"tiny": 4,   "small": 4,   "medium": 8,   "large": 12 }[args.model]

    # Pad n_heads to be divisible by world_size
    while n_heads % world != 0:
        n_heads -= 1

    # Tensor-parallel attention layer demo
    tp_attn = TensorParallelAttention(
        d_model=d_model, n_heads=n_heads, world_size=world, rank=rank
    ).to(device)

    # Benchmark tensor-parallel attention
    B, T = args.batch, 64
    x = torch.randn(B, T, d_model, device=device)

    # Warmup
    for _ in range(5):
        _ = tp_attn(x)
    if IS_DISTRIBUTED:
        dist.barrier()

    # Timed runs
    times = []
    for _ in range(args.runs):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        _ = tp_attn(x)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))

    avg_ms = sum(times) / len(times)

    if is_main():
        print(f"  Tensor-parallel attention  (world={world}):")
        print(f"    Avg latency   : {avg_ms:.3f}ms")
        print(f"    Heads/GPU     : {n_heads // world}")
        print(f"    d_model/GPU   : {d_model}")

    # ── Full single-GPU baseline for comparison ────────────────────────────────
    if is_main() and torch.cuda.device_count() >= 1:
        print(f"\n  Single-GPU baseline (TinyTransformer.generate):")
        r = run_single_gpu(args.model, args.batch, args.tokens, args.runs)
        print(f"    Avg latency   : {r['avg_ms']:.1f}ms")
        print(f"    Throughput    : {r['tps']:.1f} tok/s")

        print(f"""
  KEY OBSERVATIONS:
    1. For small models, tensor parallelism overhead (all-reduce) may not
       improve latency — the model is too fast for communication to matter.
    2. Benefits appear for large models (70B+) where:
       - Single GPU can't hold the model
       - Per-layer latency is high enough to amortise all-reduce cost
    3. NVLink bandwidth (600 GB/s on H100) vs PCIe (32 GB/s) makes a
       huge difference in whether tensor parallelism is profitable.

  NCCL PROFILING COMMANDS:
    # See NCCL all-reduce calls in nsys:
    NCCL_DEBUG=INFO nsys profile --trace=cuda,nvtx,nccl \\
        torchrun --nproc_per_node=2 infer_distributed.py

    # Benchmark NCCL all-reduce bandwidth:
    nccl-tests/build/all_reduce_perf -b 8 -e 256M -f 2 -g 2

    # Check NVLink topology:
    nvidia-smi topo -m
        """)

    if IS_DISTRIBUTED:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
