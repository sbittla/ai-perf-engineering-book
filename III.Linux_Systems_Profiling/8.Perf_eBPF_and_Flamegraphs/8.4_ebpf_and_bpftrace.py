#!/usr/bin/env python3
"""
8.Perf_eBPF_and_Flamegraphs/8.4_ebpf_and_bpftrace.py  ─  Chapter 8: eBPF and bpftrace
=======================================================================
Covers book section 8.3:
  • What eBPF is and why it is safe for production tracing
  • Key eBPF tools for AI workloads: opensnoop, biolatency, execsnoop
  • Simulating opensnoop to diagnose DataLoader file access patterns
  • Simulating biolatency I/O latency histograms
  • Essential bpftrace one-liners

Run:  python III.Linux_Systems_Profiling/8.Perf_eBPF_and_Flamegraphs/8.4_ebpf_and_bpftrace.py
All sections must print ✓.
"""

import time
import os
import tempfile
import random

print("=" * 60)
print("  Exercise 8.3 — eBPF and bpftrace")
print("=" * 60)

DEVICE = "cpu"
print(f"  Device: {DEVICE}\n")

# ─────────────────────────────────────────────────────────────
# SECTION 1: What eBPF Is
# ─────────────────────────────────────────────────────────────
print("── Section 1: What eBPF Is ──")
print("""
  eBPF (extended Berkeley Packet Filter) is a Linux kernel technology
  that lets you run small, safe programs INSIDE the kernel — without
  modifying kernel source code and without loading a kernel module.

  HOW THE SAFETY GUARANTEE WORKS:
    Before any eBPF program can execute, the kernel runs a verifier
    that checks the program exhaustively:
      • No infinite loops (all loops must terminate)
      • No out-of-bounds memory access
      • No kernel function calls that could crash the kernel
      • All registers are initialised before use
    If the verifier rejects the program, it never runs.
    If it passes, the kernel compiles it to native machine code (JIT)
    and runs it at the probe point.

  WHAT eBPF CAN TRACE:
    • Every syscall entry and exit (open, read, write, futex, …)
    • Kernel function entry and exit (any kernel function)
    • User-space function entry and exit (uprobes)
    • Network packets (the original use case)
    • Hardware performance events (PMU counters)
    • Tracepoints (stable kernel instrumentation points)

  WHY IT IS SAFE FOR PRODUCTION:
    No service restart required.  The probe attaches while the program
    runs.  Overhead is typically < 1% for moderate event rates.
    If the tracing script crashes, the kernel detaches the program
    automatically — the traced service is unaffected.

  The five most useful eBPF tools for AI infrastructure engineers:

  ┌─────────────────┬───────────────────────────────────────────────────────┐
  │ Tool            │ Purpose and typical usage                             │
  ├─────────────────┼───────────────────────────────────────────────────────┤
  │ opensnoop       │ Show every open() syscall: filename, PID, duration.   │
  │                 │ Diagnose DataLoader: what files are workers opening?  │
  ├─────────────────┼───────────────────────────────────────────────────────┤
  │ biolatency      │ Block I/O latency histogram. SSD: < 1ms. HDD: 10ms+. │
  │                 │ Diagnose: is storage causing DataLoader stalls?       │
  ├─────────────────┼───────────────────────────────────────────────────────┤
  │ execsnoop       │ Show every exec() call: new processes, their args.    │
  │                 │ Diagnose: are DataLoader workers being re-spawned?    │
  ├─────────────────┼───────────────────────────────────────────────────────┤
  │ tcplife         │ Show TCP connection lifetimes. Remote storage (S3,    │
  │                 │ NFS) connections and their transfer sizes.            │
  ├─────────────────┼───────────────────────────────────────────────────────┤
  │ funclatency     │ Latency histogram for any kernel or user function.    │
  │                 │ Profile: how long does cudaMalloc take on average?    │
  └─────────────────┴───────────────────────────────────────────────────────┘

  Installation (Ubuntu / Debian):
    sudo apt install bpfcc-tools python3-bpfcc
  or (bpftrace — more modern):
    sudo apt install bpftrace

  Running requires root:
    sudo opensnoop-bpfcc -T        # with timestamps
    sudo biolatency-bpfcc -D 10    # 10-second block I/O latency histogram
""")

def describe_ebpf_tool(name: str) -> str:
    """
    TODO 1: Implement this function.
    Return a one-line description of the named eBPF tool.
    The descriptions must contain these keywords (case-insensitive):
      "opensnoop"  → must contain "file"
      "biolatency" → must contain "latency" or "i/o"
      "execsnoop"  → must contain "exec" or "process"
      "tcplife"    → must contain "tcp" or "connection"
      "funclatency"→ must contain "function" or "latency"
    """
    descriptions = {
        "opensnoop":   "Traces every file open() syscall, showing filename, PID, and duration",
        "biolatency":  "Shows a histogram of block I/O latency to diagnose slow storage",
        "execsnoop":   "Traces every exec() call to see which processes are being spawned",
        "tcplife":     "Tracks TCP connection lifetimes, IPs, ports, and bytes transferred",
        "funclatency": "Measures latency histogram for any kernel or user-space function",
    }
    pass  # YOUR CODE HERE → return descriptions.get(name, "unknown tool")


assert describe_ebpf_tool("opensnoop") is not None, \
    "describe_ebpf_tool must return a string. Did you implement TODO 1?"
assert "file" in describe_ebpf_tool("opensnoop").lower(), (
    f"describe_ebpf_tool('opensnoop') must mention 'file'. "
    f"Got: '{describe_ebpf_tool('opensnoop')}'"
)
assert "latency" in describe_ebpf_tool("biolatency").lower() or \
       "i/o" in describe_ebpf_tool("biolatency").lower(), \
    "describe_ebpf_tool('biolatency') must mention 'latency' or 'i/o'"
assert "exec" in describe_ebpf_tool("execsnoop").lower() or \
       "process" in describe_ebpf_tool("execsnoop").lower(), \
    "describe_ebpf_tool('execsnoop') must mention 'exec' or 'process'"

print("  eBPF tool descriptions:")
for tool in ["opensnoop", "biolatency", "execsnoop", "tcplife", "funclatency"]:
    print(f"    {tool:<14}: {describe_ebpf_tool(tool)}")
print("  ✓ Section 1 passed — eBPF tool descriptions implemented")


# ─────────────────────────────────────────────────────────────
# SECTION 2: Simulating opensnoop for DataLoader Diagnosis
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: Simulating opensnoop for DataLoader Diagnosis ──")
print("""
  opensnoop-bpfcc traces every open() syscall system-wide, printing:
    PID   COMM  FD ERR PATH
    12345 python 5  0  /data/imagenet/train/n01440764/image_0001.JPEG
    12346 python 6  0  /data/imagenet/train/n02102040/image_0503.JPEG
    ...

  FOR DATALOADER DIAGNOSIS, look for:
    1. Are workers reading files SEQUENTIALLY or RANDOMLY?
       Sequential = prefetcher helps = fewer cache misses
       Random (shuffled) = no prefetching = every file is a cold read

    2. Are workers reading the SAME files repeatedly?
       Re-reading without caching = wasted I/O
       Fix: cache decoded tensors as .pt files on first epoch

    3. How many unique files are opened per second?
       Low count with high training time = few large files (OK)
       High count with many small files = lots of open() overhead

    4. Are opens spread across many worker PIDs?
       All opens from one PID = num_workers=0 (serialised loading)
       Multiple PIDs = workers running in parallel (expected)

  Run opensnoop while a DataLoader workload runs:
    # In terminal 1:
    sudo opensnoop-bpfcc -T -n python > /tmp/opens.log &

    # In terminal 2:
    python train.py --steps 100

    # Analyse:
    grep ".JPEG" /tmp/opens.log | wc -l          # total opens
    grep ".JPEG" /tmp/opens.log | sort -k4 -u | wc -l  # unique files
""")

def simulate_file_opens(n_files: int, n_reads: int) -> dict:
    """
    TODO 2: Implement this function.
    Create n_files temporary files, then perform n_reads total reads
    by cycling through all files in random order.
    Return a dict with keys:
      "unique_files"  → n_files
      "total_opens"   → n_reads
      "opens_per_file"→ n_reads // n_files

    Steps:
      1. Create n_files temporary files using tempfile.NamedTemporaryFile
         (delete=False, write b"test_data" * 100 to each)
      2. Build a list of all file paths (repeated so total accesses = n_reads)
      3. Shuffle the list
      4. Open and read each file in the shuffled order
      5. Clean up temporary files
      6. Return the result dict
    """
    pass  # YOUR CODE HERE → return result dict


result = simulate_file_opens(10, 100)
assert result is not None, "simulate_file_opens must return a dict. Did you implement TODO 2?"
assert result["total_opens"] == 100, (
    f"simulate_file_opens(10, 100)['total_opens'] should be 100, "
    f"got {result.get('total_opens')}"
)
assert result["unique_files"] == 10, \
    f"simulate_file_opens(10, 100)['unique_files'] should be 10, got {result.get('unique_files')}"
assert result["opens_per_file"] == 10, \
    f"simulate_file_opens(10, 100)['opens_per_file'] should be 10, got {result.get('opens_per_file')}"

print(f"  simulate_file_opens(10, 100):")
print(f"    unique_files  : {result['unique_files']}")
print(f"    total_opens   : {result['total_opens']}")
print(f"    opens_per_file: {result['opens_per_file']}")
print("  ✓ Section 2 passed — file open simulation matches expected counts")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Simulating biolatency for I/O Diagnosis
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: Simulating biolatency for I/O Diagnosis ──")
print("""
  biolatency-bpfcc shows a histogram of block I/O (disk) read/write
  latency.  Each bar represents how many I/O operations fell in that
  latency bucket during the measurement window.

  TYPICAL VALUES:
    NVMe SSD (PCIe 4.0): 99th percentile < 0.5ms  →  nearly all < 1ms bucket
    SATA SSD:            99th percentile < 1ms     →  most in < 1ms bucket
    Network storage NFS: 1–10ms typical, spikes 50ms+ on congestion
    HDD (rotational):    5ms average, 10–100ms tail (seek time dominates)

  INTERPRETING THE HISTOGRAM:
    If ALL I/O completes in < 1ms → storage is not the DataLoader bottleneck
    If significant I/O is in 5–20ms range → SATA SSD or network storage
    If substantial I/O is > 20ms → HDD, network storage, or storage congestion
    If spikes > 100ms → storage throttling, RAID rebuild, or NFS timeout

  biolatency command:
    sudo biolatency-bpfcc -D 10       # 10-second histogram
    sudo biolatency-bpfcc -D 10 -Q    # include QUEUE wait time
    sudo biolatency-bpfcc -D 10 -mT   # per-disk histograms with timestamps

  In this simulation, we model I/O operations with sleep() and build
  the latency histogram manually — the same four buckets biolatency shows.
""")

def build_latency_histogram(latencies_ms: list) -> dict:
    """
    TODO 3: Implement this function.
    Given a list of latency values in milliseconds, return a dict with
    counts in four buckets:
      "<1ms"   → count of values < 1.0
      "1-5ms"  → count of values >= 1.0 and < 5.0
      "5-20ms" → count of values >= 5.0 and < 20.0
      ">20ms"  → count of values >= 20.0

    All four keys must always be present (use 0 for empty buckets).
    """
    pass  # YOUR CODE HERE → return histogram dict


test_latencies = [0.5, 2.0, 8.0, 25.0]
hist = build_latency_histogram(test_latencies)
assert hist is not None, "build_latency_histogram must return a dict. Did you implement TODO 3?"
assert sum(hist.values()) == 4, (
    f"sum of histogram values should equal len(latencies_ms) = 4, "
    f"got {sum(hist.values())}"
)
assert hist["<1ms"] == 1,   f"<1ms bucket should have 1 entry (0.5ms), got {hist.get('<1ms')}"
assert hist["1-5ms"] == 1,  f"1-5ms bucket should have 1 entry (2.0ms), got {hist.get('1-5ms')}"
assert hist["5-20ms"] == 1, f"5-20ms bucket should have 1 entry (8.0ms), got {hist.get('5-20ms')}"
assert hist[">20ms"] == 1,  f">20ms bucket should have 1 entry (25.0ms), got {hist.get('>20ms')}"

# Simulate a realistic SSD access pattern and visualise it
ssd_latencies = [random.gauss(0.3, 0.15) for _ in range(80)]   # NVMe SSD
ssd_latencies += [random.gauss(2.0, 0.8) for _ in range(15)]   # occasional slower
ssd_latencies += [random.gauss(15.0, 5.0) for _ in range(5)]   # rare slow
ssd_latencies = [abs(v) for v in ssd_latencies]

ssd_hist = build_latency_histogram(ssd_latencies)
total = sum(ssd_hist.values())

print("  Simulated NVMe SSD latency histogram (100 I/O operations):")
print(f"  {'Bucket':<10}  {'Count':>6}  {'Bar'}")
max_count = max(ssd_hist.values()) if ssd_hist else 1
for bucket in ["<1ms", "1-5ms", "5-20ms", ">20ms"]:
    count = ssd_hist.get(bucket, 0)
    bar = "█" * int(count / max_count * 30)
    print(f"  {bucket:<10}  {count:>6}  {bar}")

print(f"  (Total: {total} operations)")
print("  ✓ Section 3 passed — latency histogram built and verified")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Key bpftrace One-liners
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Key bpftrace One-liners ──")
print("""
  bpftrace is a high-level tracing language for eBPF.  One-liners
  are short bpftrace programs run directly on the command line.
  They require root (sudo) but do not require kernel recompilation.

  THE 6 MOST USEFUL bpftrace ONE-LINERS FOR AI ENGINEERS:

  1. Count syscalls per second by type (is the DataLoader heavy on I/O?):
       sudo bpftrace -e 'tracepoint:raw_syscalls:sys_enter { @[args->id] = count(); }' \\
           -c "python train.py" | head -20

  2. Trace file opens with latency (which files take the longest to open?):
       sudo bpftrace -e '
       tracepoint:syscalls:sys_enter_openat { @start[tid] = nsecs; @fname[tid] = str(args->filename); }
       tracepoint:syscalls:sys_exit_openat  { @ms[str(@fname[tid])] = hist((nsecs - @start[tid])/1000000); delete(@start[tid]); delete(@fname[tid]); }
       '

  3. Histogram of read() sizes (is DataLoader doing many small reads or few large ones?):
       sudo bpftrace -e 'tracepoint:syscalls:sys_exit_read { @bytes = hist(args->ret); }'

  4. Measure futex (lock) contention latency (is Python GIL a bottleneck?):
       sudo bpftrace -e '
       tracepoint:syscalls:sys_enter_futex { @start[tid] = nsecs; }
       tracepoint:syscalls:sys_exit_futex  { @us = hist((nsecs - @start[tid]) / 1000); delete(@start[tid]); }
       '

  5. Count CUDA memory allocations over time (is allocator contention present?):
       sudo bpftrace -e 'uprobe:/usr/lib/x86_64-linux-gnu/libcuda.so:cuMemAlloc { @[comm] = count(); }'

  6. Latency distribution for any Python function (add a uprobe to python interpreter):
       sudo bpftrace -e '
       uprobe:/usr/bin/python3:PyEval_EvalFrameEx { @start[tid] = nsecs; }
       uretprobe:/usr/bin/python3:PyEval_EvalFrameEx {
           @frame_us = hist((nsecs - @start[tid]) / 1000);
           delete(@start[tid]);
       }
       '

  INSTALL bpftrace:
    sudo apt install bpftrace        # Ubuntu 22.04+
    sudo snap install bpftrace       # older Ubuntu

  VERIFY IT WORKS (no tracing, just check the tool is installed):
    sudo bpftrace --version
    sudo bpftrace -l 'tracepoint:syscalls:sys_enter_openat'
""")

def categorize_latency(ms: float) -> str:
    """
    TODO 4: Implement this function.
    Categorise an I/O latency value into a quality tier:
      ms < 0.5   → "excellent"       (NVMe SSD peak)
      ms < 2.0   → "good"            (typical NVMe SSD)
      ms < 10.0  → "acceptable"      (SATA SSD or warm cache)
      ms < 50.0  → "slow"            (HDD, network storage)
      ms >= 50.0 → "disk-bottleneck" (severe — will stall DataLoader)
    """
    pass  # YOUR CODE HERE → return tier string


assert categorize_latency(0.3)  == "excellent",        f"0.3ms → 'excellent', got '{categorize_latency(0.3)}'"
assert categorize_latency(1.0)  == "good",             f"1.0ms → 'good', got '{categorize_latency(1.0)}'"
assert categorize_latency(5.0)  == "acceptable",       f"5.0ms → 'acceptable', got '{categorize_latency(5.0)}'"
assert categorize_latency(30.0) == "slow",             f"30.0ms → 'slow', got '{categorize_latency(30.0)}'"
assert categorize_latency(75.0) == "disk-bottleneck",  f"75.0ms → 'disk-bottleneck', got '{categorize_latency(75.0)}'"
assert categorize_latency(50.0) == "disk-bottleneck",  f"50.0ms → 'disk-bottleneck', got '{categorize_latency(50.0)}'"

print("  I/O latency categories:")
for ms in [0.2, 1.5, 7.0, 35.0, 75.0]:
    print(f"    {ms:>6.1f}ms → {categorize_latency(ms)}")
print("  ✓ Section 4 passed — latency categorisation function correct")


print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 8.3 complete!")
print()
print("  You can now explain eBPF safety guarantees, describe the")
print("  five key AI engineering eBPF tools, simulate file open")
print("  patterns and I/O latency histograms, and write bpftrace")
print("  one-liners for DataLoader diagnosis.")
print()
print("  Next: III.Linux_Systems_Profiling/8.Perf_eBPF_and_Flamegraphs/8.5_lock_contention.py")
print("=" * 60)
