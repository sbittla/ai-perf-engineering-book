#!/usr/bin/env python3
"""
train.py  —  Core Training Loop (Phase 2, 3 — profiling target)
=================================================================

HOW TO RUN:
    # Basic training run:
    python train.py --task lm --steps 100 --model-size small

    # Image classification:
    python train.py --task image --steps 100 --batch 64

    # Profile with nsys:
    nsys profile --stats=true --trace=cuda,osrt \\
        python train.py --steps 50 --task lm

    # Profile with perf:
    perf record -g -F 99 python train.py --steps 50
    perf report --stdio | head -40

    # Profile with strace (syscall count):
    strace -c python train.py --steps 20

    # Flamegraph:
    py-spy record -o flamegraph.svg -- python train.py --steps 100


"""

import argparse
import sys
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import shared model definitions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'models'))
try:
    from model import TinyTransformer, SmallCNN, get_model
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'models'))
    from model import TinyTransformer, SmallCNN, get_model

# ── Argument Parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="General-purpose training script for profiling")
parser.add_argument("--task",        choices=["lm", "image"], default="lm",
                    help="Training task: lm=language model, image=image classifier")
parser.add_argument("--model-size",  choices=["tiny","small","medium","large"], default="small",
                    help="TinyTransformer size (lm task only)")
parser.add_argument("--steps",       type=int, default=100,   help="Training steps")
parser.add_argument("--batch",       type=int, default=16,    help="Batch size")
parser.add_argument("--seq-len",     type=int, default=128,   help="Sequence length (lm task)")
parser.add_argument("--lr",          type=float, default=3e-4, help="Learning rate")
parser.add_argument("--dtype",       choices=["float32","float16","bfloat16"], default="float32")
parser.add_argument("--compile",     action="store_true",     help="Apply torch.compile")
parser.add_argument("--workers",     type=int, default=2,     help="DataLoader num_workers")
parser.add_argument("--profile",     action="store_true",     help="Enable torch.profiler (5 steps)")
parser.add_argument("--checkpoint",  type=int, default=0,     help="Save checkpoint every N steps (0=off)")
parser.add_argument("--nvtx",        action="store_true",     help="Add NVTX range markers for nsys")
args = parser.parse_args()

# ── Device and dtype setup ────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
dtype = dtype_map[args.dtype]

print(f"\n{'='*60}")
print(f"  Training Script — Phase 2/3 Profiling Target")
print(f"{'='*60}")
print(f"  Task       : {args.task}")
print(f"  Device     : {device}")
if device == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU        : {props.name}  ({props.total_memory/1e9:.1f}GB)")
print(f"  Dtype      : {args.dtype}")
print(f"  Batch size : {args.batch}")
print(f"  Steps      : {args.steps}")
if args.compile:
    print(f"  Compile    : torch.compile(default)")
print(f"{'='*60}\n")

# ── Build model and dataset ───────────────────────────────────────────────────
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

if args.task == "lm":
    # ── Language Modelling ─────────────────────────────────────────────────
    # Input: random token IDs  Output: next-token logits
    # Loss: cross-entropy over vocabulary

    model = TinyTransformer(size=args.model_size, max_seq=args.seq_len + 1)
    print(f"  Model: TinyTransformer [{args.model_size}]  params={model.param_count()}")
    mem = TinyTransformer.estimate_memory_gb(args.model_size,
                                             batch=args.batch, seq=args.seq_len)
    print(f"  Estimated training memory: {mem['total_train_gb']:.2f} GB")

    # Synthetic token dataset
    # In production: replace with a real tokenized text corpus
    n_samples = args.batch * args.steps * 2
    tokens = torch.randint(0, 50257, (n_samples, args.seq_len + 1))
    dataset = TensorDataset(tokens)
    # Input = tokens[:, :-1], Target = tokens[:, 1:] (next-token prediction)

    criterion = nn.CrossEntropyLoss()

else:
    # ── Image Classification ────────────────────────────────────────────────
    # Input: random float images (B, 3, 224, 224)  Output: class logits

    model = SmallCNN(num_classes=1000)
    print(f"  Model: SmallCNN  params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # Synthetic image dataset
    n_samples = args.batch * args.steps * 2
    images = torch.rand(n_samples, 3, 224, 224)
    labels = torch.randint(0, 1000, (n_samples,))
    dataset = TensorDataset(images, labels)

    criterion = nn.CrossEntropyLoss()

# Move model to device and optionally convert dtype
model = model.to(device=device, dtype=dtype if dtype != torch.float16 else torch.float32)
# Note: float16 training needs AMP (see below) — model weights stay float32

# Optional torch.compile
if args.compile:
    print(f"  Compiling model with torch.compile...")
    model = torch.compile(model, mode="default")
    print(f"  Compilation staged (triggers on first forward pass)")

model.train()

# DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=args.batch,
    shuffle=True,
    num_workers=args.workers,
    pin_memory=(device == "cuda"),   # Faster H2D transfers
    drop_last=True,                  # Consistent batch sizes for profiling
)

optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

# AMP scaler: Automatic Mixed Precision uses FP16 for forward/backward
# but keeps FP32 master weights. Speeds up compute, halves activation memory.
# The GradScaler prevents FP16 underflow in gradients.
use_amp  = (device == "cuda") and (args.dtype in ["float16", "bfloat16"])
scaler   = torch.cuda.amp.GradScaler(enabled=use_amp)
amp_dtype = dtype if use_amp else torch.float32

print(f"  AMP: {'enabled (' + args.dtype + ')' if use_amp else 'disabled (float32)'}")
print(f"  DataLoader workers: {args.workers}\n")

# ── Optional: torch.profiler context ─────────────────────────────────────────
if args.profile:
    from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
    prof_schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
    profiler = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_schedule,
        on_trace_ready=tensorboard_trace_handler("logs/tb_train"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )
    profiler.__enter__()
    print("  torch.profiler enabled (logs/tb_train) — will run 5 steps then stop\n")

# ── Training loop ─────────────────────────────────────────────────────────────
step_times  = []
loss_values = []
warmup_done = False

total_start = time.perf_counter()

for step, batch in enumerate(dataloader):
    if step >= args.steps:
        break

    start_evt = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
    end_evt   = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
    wall_start = time.perf_counter()

    if start_evt:
        start_evt.record()

    # ── NVTX markers: named ranges visible in Nsight Systems timeline ────────
    # Range names appear as coloured bands in the nsys-ui timeline view.
    # This makes it easy to identify phases: data prep vs forward vs backward.
    if args.nvtx:
        torch.cuda.nvtx.range_push(f"step_{step}")

    # ── Data preparation ──────────────────────────────────────────────────────
    if args.nvtx:
        torch.cuda.nvtx.range_push("data_transfer")

    if args.task == "lm":
        tokens = batch[0].to(device, non_blocking=True)
        input_ids = tokens[:, :-1]    # (B, T) — input tokens
        targets   = tokens[:, 1:]     # (B, T) — next-token targets
    else:
        images = batch[0].to(device, non_blocking=True)
        labels = batch[1].to(device, non_blocking=True)

    if args.nvtx:
        torch.cuda.nvtx.range_pop()   # end data_transfer

    # ── Forward pass ──────────────────────────────────────────────────────────
    if args.nvtx:
        torch.cuda.nvtx.range_push("forward")

    optimiser.zero_grad(set_to_none=True)

    # torch.autocast applies AMP: matmuls run in FP16, accumulation in FP32
    with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
        if args.task == "lm":
            logits, _ = model(input_ids)
            # Flatten for cross-entropy: (B, T, V) → (B*T, V) vs (B*T,)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        else:
            out  = model(images)
            loss = criterion(out, labels)

    if args.nvtx:
        torch.cuda.nvtx.range_pop()   # end forward

    # ── Backward pass ─────────────────────────────────────────────────────────
    if args.nvtx:
        torch.cuda.nvtx.range_push("backward")

    # scaler.scale multiplies the loss by a scale factor to prevent FP16 underflow
    # scaler.step unscales gradients and calls optimiser.step
    # scaler.update adjusts the scale factor for next step
    scaler.scale(loss).backward()
    scaler.unscale_(optimiser)
    # Gradient clipping: prevent exploding gradients (important for transformers)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimiser)
    scaler.update()

    if args.nvtx:
        torch.cuda.nvtx.range_pop()   # end backward
        torch.cuda.nvtx.range_pop()   # end step_N

    # ── Timing ────────────────────────────────────────────────────────────────
    if end_evt:
        end_evt.record()
        torch.cuda.synchronize()
        step_ms = start_evt.elapsed_time(end_evt)
    else:
        step_ms = (time.perf_counter() - wall_start) * 1000

    step_times.append(step_ms)

    # Only call .item() every 10 steps to avoid per-step GPU barriers
    if step % 10 == 0:
        loss_val = loss.item()   # GPU sync — acceptable here (only every 10 steps)
        loss_values.append(loss_val)
        avg_ms = sum(step_times[-10:]) / min(10, len(step_times))
        print(f"  step {step:4d}/{args.steps}  loss={loss_val:.4f}  "
              f"step_ms={avg_ms:.1f}  ", end="")
        if device == "cuda":
            mem_gb = torch.cuda.memory_allocated() / 1e9
            print(f"vram={mem_gb:.2f}GB")
        else:
            print()

    # ── Optional checkpoint save ──────────────────────────────────────────────
    if args.checkpoint > 0 and (step + 1) % args.checkpoint == 0:
        ckpt_path = f"checkpoints/step_{step+1}.pt"
        torch.save({
            "step": step,
            "model_state": model.state_dict(),
            "optimiser_state": optimiser.state_dict(),
            "loss": loss.item(),
        }, ckpt_path)
        print(f"  → Saved checkpoint: {ckpt_path}")

    # ── torch.profiler step ───────────────────────────────────────────────────
    if args.profile:
        profiler.step()
        if step >= 4:   # wait(1) + warmup(1) + active(3) = 5 total
            break

if args.profile:
    profiler.__exit__(None, None, None)
    print(f"\n  torch.profiler output: logs/tb_train/")
    print(f"  View: tensorboard --logdir logs/tb_train")

# ── Summary ───────────────────────────────────────────────────────────────────
total_time = time.perf_counter() - total_start
n_steps = min(args.steps, step + 1)
avg_step = sum(step_times) / len(step_times) if step_times else 0

print(f"\n{'='*60}")
print(f"  TRAINING COMPLETE")
print(f"{'='*60}")
print(f"  Steps completed : {n_steps}")
print(f"  Total time      : {total_time:.1f}s")
print(f"  Avg step time   : {avg_step:.1f}ms")
print(f"  Steps/sec       : {n_steps/total_time:.1f}")
if device == "cuda":
    print(f"  Peak GPU memory : {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
print(f"{'='*60}")
