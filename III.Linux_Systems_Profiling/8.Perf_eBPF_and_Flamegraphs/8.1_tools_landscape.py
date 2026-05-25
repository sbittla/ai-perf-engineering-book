"""
Exercise 8.1 — The Linux Performance Toolchain: Origins and Landscape

Chapter 8 (Background): Where These Profiling Tools Came From
Book: AI Systems Performance Engineering

Run:
    python 8.1_tools_landscape.py

What this exercise does:
  1. Checks which Linux performance tools are installed on this system.
  2. Runs a cProfile demo on a CPU workload and explains how to read the output.
  3. Explains perf and the perf_event_open() syscall (hardware PMU counters).
  4. Describes how flamegraphs were invented and how to generate them with py-spy.
  5. Covers eBPF safety guarantees, BCC tools, and bpftrace one-liners.
  6. Provides a symptom-to-tool diagnostic matrix for common AI workload issues.

No TODOs here — this is a read-and-run exercise. Read each section's output,
understand what it means, then explore the chapters in Part III for hands-on work.
"""

import sys
import os
import subprocess
import time
import cProfile
import pstats
import io
import math

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Tool Availability Check
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 1: Tool Availability on This System")
print("=" * 60)

tools = {
    "perf"       : ("Hardware PMU counters, CPU cycles, cache misses",
                    "sudo perf stat -e cycles,cache-misses python train.py"),
    "py-spy"     : ("Python sampling profiler, zero-overhead flamegraphs",
                    "sudo py-spy top --pid <PID>"),
    "bpftrace"   : ("eBPF one-liners for kernel-level tracing",
                    "sudo bpftrace -e 'tracepoint:sched:sched_switch { @[comm] = count(); }'"),
    "nsys"       : ("NVIDIA Nsight Systems — GPU/CPU timeline profiler",
                    "nsys profile --trace=cuda,nvtx python train.py"),
    "ncu"        : ("NVIDIA Nsight Compute — per-kernel roofline analysis",
                    "ncu --set full -o report python train.py"),
    "strace"     : ("System call tracer — see every syscall a process makes",
                    "strace -c -p <PID>"),
    "lsof"       : ("List open file descriptors — diagnose I/O bottlenecks",
                    "lsof -p <PID> | grep -v mem"),
}

print(f"  {'Tool':<12} {'Available':<12} {'Purpose'}")
print(f"  {'-'*12} {'-'*12} {'-'*40}")
for tool, (purpose, _) in tools.items():
    found = subprocess.run(["which", tool], capture_output=True).returncode == 0
    status = "YES" if found else "not found"
    print(f"  {tool:<12} {status:<12} {purpose}")

print()
print("Missing tools can usually be installed with:")
print("  sudo apt-get install linux-perf bpftrace strace lsof")
print("  pip install py-spy")
print("  (Nsight tools require NVIDIA driver + CUDA toolkit)")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: cProfile — The Built-in Python Profiler
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 2: cProfile — Python's Built-in Profiler")
print("=" * 60)

print("""
cProfile instruments every function call at the interpreter level.
It is the first tool to reach for when you don't know where Python
is spending its time. It has no external dependencies.

How it works:
  - CPython's eval loop calls a C hook on every function entry/exit.
  - cProfile records call count and cumulative time per function.
  - Overhead: ~10-30% depending on call density.

Limitation: it only sees Python time. GPU kernels running asynchronously
are invisible to cProfile — use Nsight Systems for GPU work.
""")

def fib_recursive(n):
    if n <= 1:
        return n
    return fib_recursive(n - 1) + fib_recursive(n - 2)

def matrix_naive(n):
    A = [[float(i + j) for j in range(n)] for i in range(n)]
    B = [[float(i * j + 1) for j in range(n)] for i in range(n)]
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            for j in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

def workload():
    fib_recursive(28)
    matrix_naive(60)

pr = cProfile.Profile()
pr.enable()
workload()
pr.disable()

stream = io.StringIO()
ps = pstats.Stats(pr, stream=stream).sort_stats("cumulative")
ps.print_stats(8)
print(stream.getvalue())

print("Reading the output above:")
print("  ncalls  — how many times this function was called")
print("  tottime — time spent IN this function (excluding callees)")
print("  cumtime — total time including all callees")
print("  The function with the highest tottime is usually the bottleneck.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Section 3: perf — Hardware Performance Counters
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 3: perf — Hardware PMU Counters")
print("=" * 60)

print("""
'perf' speaks directly to the CPU's Performance Monitoring Unit (PMU).
Every modern CPU has dedicated hardware registers that count events at
zero cost to the running program — no code modification needed.

Key counters for AI workloads:

  cycles          — raw clock ticks consumed
  instructions    — retired instructions (IPC = instructions/cycles)
  cache-misses    — L3 misses that go to DRAM (very expensive)
  branch-misses   — mispredicted branches causing pipeline flushes
  page-faults     — virtual memory page allocations

IPC (Instructions Per Cycle) is the most important single number:
  IPC < 1.0  → memory-bound (CPU is stalling waiting for data)
  IPC 1-2    → mixed compute/memory
  IPC > 2    → compute-bound (pipeline is well-fed)

Example command:
  sudo perf stat -e cycles,instructions,cache-misses,branch-misses \\
       python train.py

How 'perf' works internally (2009 kernel addition):
  - perf_event_open() syscall registers a counter with the kernel.
  - The kernel programs the PMU hardware registers on each CPU core.
  - On overflow, the PMU fires an interrupt; the kernel records the
    call stack at that moment (sampling mode) or accumulates counts.
  - Requires CAP_PERFMON or sudo on most systems.
""")

# Demonstrate that perf is a syscall-based tool
print("The perf_event_open syscall number on x86_64 is 298.")
print("You can verify: strace -e perf_event_open perf stat sleep 1")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Flamegraphs — Visualising Where Time Goes
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 4: Flamegraphs")
print("=" * 60)

print("""
A flamegraph turns a stream of sampled call stacks into a visual:

  ┌─────────────────────────────────────────────────────────┐
  │ torch.nn.Module.forward                                  │ (wide = slow)
  │   ┌────────────────────────────┐ ┌──────────────────┐   │
  │   │ nn.Linear.forward          │ │ nn.ReLU.forward   │   │
  │   │   ┌────────────────────┐   │ │                   │   │
  │   │   │ F.linear           │   │ │                   │   │
  │   │   │   ┌────────────┐   │   │ │                   │   │
  │   │   │   │cublasSgemm │   │   │ │                   │   │
  │   │   │   └────────────┘   │   │ │                   │   │
  │   │   └────────────────────┘   │ │                   │   │
  │   └────────────────────────────┘ └──────────────────┘   │
  └─────────────────────────────────────────────────────────┘
  Width = time. Deep call stacks appear tall. Hottest code is wide at the bottom.

Brendan Gregg invented flamegraphs in 2011 while debugging a MySQL
performance issue. Traditional profiler output (sorted tables) made it
hard to see the COMPOSITION of where time went. Flamegraphs solve this.

For Python processes, py-spy captures stacks without pausing the process:
  sudo py-spy record -o profile.svg --pid <PID>
  sudo py-spy top --pid <PID>           (htop-style live view)

For PyTorch specifically, torch.profiler wraps Nsight data and can
export Chrome traces viewable at chrome://tracing or perfetto.dev.
""")

# ─────────────────────────────────────────────────────────────────────────────
# Section 5: eBPF — Safe Kernel Tracing Without Modules
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 5: eBPF — The Modern Observability Superpower")
print("=" * 60)

print("""
Before eBPF: if you needed to trace a kernel event (e.g., every file open,
every TCP packet, every context switch), you had to write a kernel MODULE.
Kernel modules run with full privileges — one bug causes a kernel panic
and system crash. They also require recompiling for every kernel version.

eBPF (2014, Linux 3.18+) changed this:
  - You write a small program in restricted C.
  - The kernel verifier proves it is safe (no infinite loops, bounded memory).
  - The JIT compiler converts it to native machine code.
  - It attaches to a kernel event (tracepoint, kprobe, uprobe, socket).
  - It runs at kernel speed with near-zero overhead.
  - No reboot, no risk of kernel panic.

Key eBPF tools for AI infrastructure:

  BCC (BPF Compiler Collection):
    opensnoop          — trace every file open() call systemwide
    biolatency         — histogram of disk I/O latency
    execsnoop          — trace every exec() (new process)
    profile            — CPU profiler using hardware PMU sampling

  bpftrace (one-liners):
    # Count context switches per process name
    sudo bpftrace -e 'tracepoint:sched:sched_switch { @[comm] = count(); }'

    # Histogram of read() syscall duration
    sudo bpftrace -e 'tracepoint:syscalls:sys_enter_read { @ts[tid] = nsecs; }
                      tracepoint:syscalls:sys_exit_read  /@ts[tid]/
                      { @ms = hist((nsecs - @ts[tid]) / 1e6); delete(@ts[tid]); }'

Use case for AI training:
  If your DataLoader workers are slow, eBPF can tell you exactly which
  file is taking longest to open, and whether it is I/O or CPU bound —
  without adding a single line of instrumentation to your code.
""")

# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Diagnostic Tool Matrix
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 6: Symptom → Tool Mapping")
print("=" * 60)

matrix = [
    ("GPU util < 30%",        "py-spy / cProfile",    "CPU bottleneck on Python side"),
    ("GPU util 100%, slow",   "Nsight Systems (nsys)", "Memory bandwidth or kernel efficiency"),
    ("High IPC, slow",        "Nsight Compute (ncu)",  "Roofline: compute-bound kernel"),
    ("Low IPC (< 1.0)",       "perf cache-misses",     "Memory-bound: cache misses"),
    ("DataLoader slow",       "py-spy + bpftrace",     "File I/O or CPU preprocessing"),
    ("Random OOM crashes",    "nvidia-smi dmon",       "Memory fragmentation over time"),
    ("Spiky P99 latency",     "perf + flamegraph",     "GC pauses or lock contention"),
    ("Kernel panic / crash",  "dmesg + journalctl",    "Driver bug or OOM killer"),
]

print(f"  {'Symptom':<28} {'Tool':<24} {'Likely Cause'}")
print(f"  {'-'*28} {'-'*24} {'-'*35}")
for symptom, tool, cause in matrix:
    print(f"  {symptom:<28} {tool:<24} {cause}")

print()
print("Explore next: III.Linux_Systems_Profiling/8.Perf_eBPF_and_Flamegraphs/")
print("              III.Linux_Systems_Profiling/9.Memory_Hierarchy_and_NUMA/")
