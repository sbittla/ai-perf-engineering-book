#!/usr/bin/env python3
"""
B.1_profiling_cheatsheet.py  —  Appendix B: Command Reference

An interactive reference that:
  1. Prints the diagnostic decision tree (which tool to use first)
  2. Demonstrates torch.profiler with record_function on a small model
  3. Demonstrates CUDA event timing vs wall-clock timing
  4. Prints the full command cheatsheet grouped by tool

Run:  python B.1_profiling_cheatsheet.py
"""

import sys, time
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function

print("=" * 68)
print("  APPENDIX B — Profiling Command Reference")
print("=" * 68)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n  Device: {DEVICE}\n")

# ──────────────────────────────────────────────────────────────────────────────
# Section 1 — Diagnostic Decision Tree
# ──────────────────────────────────────────────────────────────────────────────
print("─" * 68)
print("Section 1 — Diagnostic Decision Tree")
print("─" * 68)

DECISION_TREE = """
CONCEPT: Use the right tool for the right question

  SYMPTOM                         → TOOL
  ─────────────────────────────────────────────────────────────────────
  "Is the GPU busy at all?"       → nvidia-smi (live util %)
  "Where is time going?"          → nsys profile --stats=true
  "Which kernel is the hotspot?"  → torch.profiler  (PyTorch ops)
  "Is this kernel memory-bound?"  → ncu --metrics dram__throughput
  "Why is the CPU slow?"          → py-spy record (CPU flamegraph)
  "Is disk I/O starving the GPU?" → iostat -xz 1 && vmstat 1
  "Is there NUMA penalty?"        → numastat && numactl --hardware
  "Is NCCL the bottleneck?"       → nsys --trace=cuda,nvtx,nccl

  QUICK DIAGNOSTIC ORDER:
    1.  watch -n 0.5 nvidia-smi            # check GPU util %
    2.  nsys profile --stats=true ...       # timeline view
    3.  ncu --set basic ...                 # per-kernel view
    4.  py-spy record -o flame.svg -- ...  # CPU flamegraph
    5.  iostat -xz 1 && vmstat 1           # I/O + iowait
"""
print(DECISION_TREE)

# TODO: Print the two most common mistakes to verify you've absorbed them
COMMON_MISTAKES = [
    ("No warmup",         "First call triggers cuBLAS autotuning → always slow"),
    ("time.time() on GPU","Measures CPU submission, not GPU execution → wrong"),
    ("Single sample",     "No variance, no P99 — always run 20+ iterations"),
    ("Mixed variables",   "Changing batch AND precision together → can't isolate cause"),
    ("Reporting peak",    "Peak ≠ sustained — GPU throttles under thermal pressure"),
]
print("  The 5 Benchmarking Mistakes (Chapter 14):")
for i, (name, desc) in enumerate(COMMON_MISTAKES, 1):
    print(f"    {i}. {name:<22}  {desc}")

print("\n✓ Section 1 passed\n")

# ──────────────────────────────────────────────────────────────────────────────
# Section 2 — torch.profiler Live Demo
# ──────────────────────────────────────────────────────────────────────────────
print("─" * 68)
print("Section 2 — torch.profiler Live Demo")
print("─" * 68)

print("""
CONCEPT: torch.profiler records CPU ops AND CUDA kernels together.
  Use record_function() to annotate your own regions.
  Sort by self_cuda_time_total to find the real hotspot.
""")

class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8192, 256)
        self.ln    = nn.LayerNorm(256)
        self.fc1   = nn.Linear(256, 1024)
        self.fc2   = nn.Linear(1024, 256)
        self.head  = nn.Linear(256, 8192)

    def forward(self, idx):
        with record_function("embedding"):
            x = self.embed(idx)
        with record_function("norm"):
            x = self.ln(x)
        with record_function("ffn"):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
        with record_function("lm_head"):
            return self.head(x)

model = SmallModel().to(DEVICE).eval()
idx   = torch.randint(0, 8192, (4, 64), device=DEVICE)

# Warmup
with torch.no_grad():
    for _ in range(3):
        _ = model(idx)

# TODO: Run torch.profiler and print the top-5 ops by CUDA/CPU time
activities = [ProfilerActivity.CPU]
if DEVICE == "cuda":
    activities.append(ProfilerActivity.CUDA)

with profile(activities=activities, record_shapes=True) as prof:
    with torch.no_grad():
        for _ in range(5):
            _ = model(idx)

if DEVICE == "cuda":
    sort_key = "self_cuda_time_total"
else:
    sort_key = "self_cpu_time_total"

print(f"  torch.profiler top-5 ops (sorted by {sort_key}):\n")
print(prof.key_averages().table(sort_by=sort_key, row_limit=5))

# Confirm named regions appear in the trace
region_names = {evt.key for evt in prof.key_averages()}
found_regions = {"embedding", "ffn", "lm_head"} & region_names
assert len(found_regions) > 0, (
    f"Expected named regions in profiler output, found: {region_names}"
)
print(f"  Named regions captured: {sorted(found_regions)}")

print("\n✓ Section 2 passed\n")

# ──────────────────────────────────────────────────────────────────────────────
# Section 3 — CUDA Events vs wall-clock timing
# ──────────────────────────────────────────────────────────────────────────────
print("─" * 68)
print("Section 3 — CUDA Events vs Wall-Clock Timing")
print("─" * 68)

print("""
CONCEPT: time.perf_counter() on GPU code measures CPU submission time, not
  GPU execution time. CUDA events bracket the GPU timeline directly.

  CORRECT pattern:
    torch.cuda.synchronize()        # flush any pending work
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt   = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    <GPU work here>
    end_evt.record()
    torch.cuda.synchronize()
    elapsed_ms = start_evt.elapsed_time(end_evt)
""")

# TODO: Time a matmul both ways — show the difference (or similarity on CPU)
SIZE = 2048
a = torch.randn(SIZE, SIZE, device=DEVICE)
b = torch.randn(SIZE, SIZE, device=DEVICE)
N = 20

# Wall-clock timing (wrong for GPU)
if DEVICE == "cuda":
    torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    c = torch.mm(a, b)
if DEVICE == "cuda":
    torch.cuda.synchronize()
wall_ms = (time.perf_counter() - t0) / N * 1000

# CUDA event timing (correct)
if DEVICE == "cuda":
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(N):
        c = torch.mm(a, b)
    e.record()
    torch.cuda.synchronize()
    cuda_ms = s.elapsed_time(e) / N
else:
    t0 = time.perf_counter()
    for _ in range(N):
        c = torch.mm(a, b)
    cuda_ms = (time.perf_counter() - t0) / N * 1000

print(f"  {SIZE}×{SIZE} matmul on {DEVICE}:")
print(f"    wall-clock time   : {wall_ms:.3f} ms")
print(f"    CUDA event time   : {cuda_ms:.3f} ms")
if DEVICE == "cuda":
    ratio = wall_ms / cuda_ms if cuda_ms > 0 else 1.0
    note  = "close (overhead amortised)" if ratio < 1.2 else "wall-clock was INFLATED"
    print(f"    ratio             : {ratio:.2f}x  ({note})")
else:
    print("    [CPU: both methods equivalent — CUDA events require GPU]")

assert wall_ms > 0 and cuda_ms > 0, "Timing error — got zero elapsed time"

print("\n✓ Section 3 passed\n")

# ──────────────────────────────────────────────────────────────────────────────
# Section 4 — Command Reference Tables
# ──────────────────────────────────────────────────────────────────────────────
print("─" * 68)
print("Section 4 — Command Reference (print-ready cheatsheet)")
print("─" * 68)

CHEATSHEET = """
┌─────────────────────────────────────────────────────────────────────┐
│  NSIGHT SYSTEMS  (nsys)  — GPU timeline, ms granularity             │
├─────────────────────────────────────────────────────────────────────┤
│  nsys profile --stats=true python train.py                          │
│  nsys profile -o report --trace=cuda,nvtx,osrt python train.py     │
│  nsys profile --delay=5 --duration=10 -o snap python serve.py      │
│  Look for: white GPU gaps (CPU bound), long H2D bars (pin_memory)  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  NSIGHT COMPUTE  (ncu)  — kernel counters, nanosecond granularity   │
├─────────────────────────────────────────────────────────────────────┤
│  ncu --set basic python matmul_bench.py                             │
│  ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,  │
│               sm__throughput.avg.pct_of_peak_sustained_elapsed      │
│               --kernel-name ".*mm.*" python script.py               │
│  Rule: dram≈100% → memory-bound; sm≈100% → compute-bound           │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  TORCH.PROFILER  — PyTorch op granularity                           │
├─────────────────────────────────────────────────────────────────────┤
│  with profile(activities=[CPU, CUDA], record_shapes=True) as p:     │
│      model(x)                                                       │
│  print(p.key_averages().table(sort_by="self_cuda_time_total"))      │
│  p.export_chrome_trace("trace.json")   # open in chrome://tracing   │
│  python -m torch.utils.bottleneck script.py  # quick combined view  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  NVIDIA-SMI  — live GPU utilisation                                 │
├─────────────────────────────────────────────────────────────────────┤
│  watch -n 0.5 nvidia-smi                                            │
│  nvidia-smi --query-gpu=util.gpu,memory.used --format=csv -l 1      │
│  nvidia-smi dmon -s mu -d 1                                         │
│  nvitop -m full                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  PY-SPY  — Python CPU flamegraphs (no sudo needed)                  │
├─────────────────────────────────────────────────────────────────────┤
│  py-spy record -o flame.svg -- python train.py                      │
│  py-spy record -o before.folded --format raw -- python slow.py      │
│  py-spy record -o after.folded  --format raw -- python fast.py      │
│  ~/FlameGraph/difffolded.pl before.folded after.folded \\           │
│      | ~/FlameGraph/flamegraph.pl > diff.svg                        │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  LINUX PERF  — hardware events                                      │
├─────────────────────────────────────────────────────────────────────┤
│  perf stat -e cycles,instructions,cache-misses python train.py      │
│  perf record -g -F 99 python train.py && perf report --stdio        │
│  perf script | stackcollapse-perf.pl | flamegraph.pl > cpu.svg      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  EBPF / BCC TOOLS  — kernel-level tracing                           │
├─────────────────────────────────────────────────────────────────────┤
│  opensnoop -p $(pgrep python)      # file opens                     │
│  biolatency -D                      # block I/O latency histogram    │
│  runqlat                            # scheduler latency              │
│  bpftrace -e 'tracepoint:syscalls:sys_enter_read { @[comm]=count;}' │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  I/O + SYSTEM MONITORING                                            │
├─────────────────────────────────────────────────────────────────────┤
│  vmstat 1 30             # CPU states, memory, swap (wa = iowait)   │
│  iostat -xz 1            # disk throughput + utilisation            │
│  sar -u 1 60             # CPU utilisation history                  │
│  iowait > 20%  → disk bottleneck; swap si/so > 0 → memory pressure │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  NUMA + CPU AFFINITY                                                │
├─────────────────────────────────────────────────────────────────────┤
│  numactl --hardware                                                  │
│  numactl --cpunodebind=0 --membind=0 python train.py                │
│  taskset -c 0-7 python train.py                                     │
│  cat /sys/bus/pci/devices/0000:01:00.0/numa_node   # GPU NUMA node  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  PYTORCH MEMORY                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  torch.cuda.memory_allocated() / 1e9      # GB in use               │
│  torch.cuda.max_memory_allocated() / 1e9  # peak since reset        │
│  torch.cuda.reset_peak_memory_stats()     # reset peak counter      │
│  torch.cuda.empty_cache()                 # release cached blocks   │
│  print(torch.cuda.memory_summary())       # full breakdown           │
└─────────────────────────────────────────────────────────────────────┘
"""
print(CHEATSHEET)

print("✓ Section 4 passed\n")

# ──────────────────────────────────────────────────────────────────────────────
print("=" * 68)
print("  ALL SECTIONS PASSED")
print(f"  Profiling cheatsheet complete (device={DEVICE})")
print()
print("  Next: Appendix C — C.1_interview_questions.py")
print("=" * 68)
