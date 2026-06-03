#!/usr/bin/env python3
"""
9.Memory_Hierarchy_and_NUMA/9.2_numa_and_topology.py  ─  Chapter 9: NUMA Architecture
=======================================================================
Covers book section 9.2:
  • What NUMA is and why it matters on multi-socket AI servers
  • Detecting NUMA topology with numactl --hardware
  • CPU affinity — reading and restricting the allowed core set
  • numactl binding commands for GPU training
  • Simulating local vs remote NUMA memory access latency

Run:  python III.Linux_Systems_Profiling/9.Memory_Hierarchy_and_NUMA/9.2_numa_and_topology.py
All sections must print ✓.
"""

import os
import re
import subprocess
import time
import numpy as np

print("=" * 60)
print("  Exercise 9.2 — NUMA Architecture and Topology")
print("=" * 60)

DEVICE = "cpu"
print(f"  Device: {DEVICE}\n")

# ─────────────────────────────────────────────────────────────
# SECTION 1: NUMA Topology Detection
# ─────────────────────────────────────────────────────────────
print("── Section 1: NUMA Topology Detection ──")
print("""
  NUMA (Non-Uniform Memory Access) describes a memory architecture
  where memory is divided into banks, each bank directly attached to
  one CPU socket.  Accessing local memory (same socket) is fast.
  Accessing remote memory (different socket) must cross the socket
  interconnect — Intel calls it UPI (Ultra Path Interconnect),
  AMD calls it Infinity Fabric.

  REMOTE MEMORY PENALTY:
    Typical latency increase: 1.5× to 2.5× slower.
    For sequential streaming bandwidth: 10–30% reduction (bandwidth-
    limited operations recover some of the latency penalty through
    pipelining, but not all).

  WHY IT MATTERS FOR AI WORKLOADS:
    A typical 2-socket server has 2 NUMA nodes.
    GPU 0 is connected to the PCIe root complex of socket 0.
    If your DataLoader workers are pinned to socket 1 (the "wrong"
    socket), the data they load must travel:
      Socket 1 DRAM → UPI interconnect → Socket 0 DRAM → PCIe → GPU
    instead of the direct path:
      Socket 0 DRAM → PCIe → GPU

    This adds 10–20% to H2D transfer time on every batch.
    In a pipeline running thousands of batches, this compounds.

  DETECTING YOUR NUMA TOPOLOGY:
    numactl --hardware    (show nodes, CPUs, memory per node, distances)
    lstopo --of ascii     (full topology: CPU + cache + NUMA + GPU)
    cat /proc/cpuinfo | grep "physical id" | sort -u | wc -l  (socket count)

  For this exercise, we parse simulated numactl output
  (the function works on real output too).
""")

def parse_numa_nodes(numactl_output: str) -> int:
    """
    TODO 1: Implement this function.
    Parse the output of `numactl --hardware` and return the number
    of NUMA nodes found.

    Look for lines that start with "node N" where N is a digit,
    but EXCLUDE the "node distances:" header line (no digit after it).

    Example input:
      "node 0 cpus: 0-15\\nnode 1 cpus: 16-31\\nnode distances:\\n"

    Approach:
      Use re.findall(r"^node (\\d+) cpus:", numactl_output, re.MULTILINE)
      The length of the match list is the node count.
    """
    return len(re.findall(r"^node (\d+) cpus:", numactl_output, re.MULTILINE))


# Test with synthetic numactl output
sample_output_2node = (
    "available: 2 nodes (0-1)\n"
    "node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15\n"
    "node 0 size: 128614 MB\n"
    "node 0 free: 98432 MB\n"
    "node 1 cpus: 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31\n"
    "node 1 size: 128614 MB\n"
    "node 1 free: 97820 MB\n"
    "node distances:\n"
    "node   0   1\n"
    "  0:  10  21\n"
    "  1:  21  10\n"
)

sample_output_4node = (
    "available: 4 nodes (0-3)\n"
    "node 0 cpus: 0 1 2 3\n"
    "node 1 cpus: 4 5 6 7\n"
    "node 2 cpus: 8 9 10 11\n"
    "node 3 cpus: 12 13 14 15\n"
    "node distances:\n"
    "node   0   1   2   3\n"
)

result_2 = parse_numa_nodes(sample_output_2node)
result_4 = parse_numa_nodes(sample_output_4node)

assert result_2 is not None, "parse_numa_nodes must return an int. Did you implement TODO 1?"
assert result_2 == 2, (
    f"parse_numa_nodes with 2 'node N cpus:' lines → 2, got {result_2}"
)
assert result_4 == 4, (
    f"parse_numa_nodes with 4 'node N cpus:' lines → 4, got {result_4}"
)

# Try real numactl (graceful fallback if not installed)
numactl_real = ""
try:
    result = subprocess.run(
        ["numactl", "--hardware"],
        capture_output=True, text=True, timeout=5
    )
    if result.returncode == 0:
        numactl_real = result.stdout
        real_nodes = parse_numa_nodes(numactl_real)
        print(f"  Live numactl output detected {real_nodes} NUMA node(s) on this system.")
    else:
        print("  numactl not available — using synthetic output for testing.")
except (FileNotFoundError, subprocess.TimeoutExpired):
    print("  numactl not installed — using synthetic output for testing.")

print(f"  parse_numa_nodes(2-node sample) = {result_2}")
print(f"  parse_numa_nodes(4-node sample) = {result_4}")
print("  ✓ Section 1 passed — NUMA node parsing implemented")


# ─────────────────────────────────────────────────────────────
# SECTION 2: CPU Affinity
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: CPU Affinity ──")
print("""
  CPU affinity is the set of CPU cores this process is ALLOWED to run on.
  The OS scheduler only places the process on cores in its affinity set.

  READING AFFINITY:
    Python:  os.sched_getaffinity(0)   → set of core IDs
    Shell:   taskset -c -p $$           → comma-separated or range

  SETTING AFFINITY (restricts the process):
    Python:  os.sched_setaffinity(0, {0, 1, 2, 3})
    Shell:   taskset -c 0-3 python train.py
    numactl: numactl --cpunodebind=0 python train.py  (all cores on node 0)

  WHY AFFINITY MATTERS FOR AI TRAINING:
    On a 2-socket server, socket 0 has cores 0–15 and socket 1 has
    cores 16–31 (typical example).  GPU 0's PCIe root complex connects
    to socket 0.  If your training process runs freely on all 32 cores,
    the OS may schedule it on socket 1 cores — every memory access is
    then "remote" relative to GPU 0.

    Pinning to socket 0 cores ensures:
      1. All memory allocated by the training process sits on socket 0
         DRAM (first-touch allocation policy).
      2. All PCIe DMA transfers from this process go directly over the
         socket 0 PCIe lanes — no cross-socket hop.

  NUMACTL STRATEGY FOR MULTI-GPU SERVERS:
    # Find which NUMA node your GPU is on:
    cat /sys/bus/pci/devices/*/numa_node

    # Run training bound to that GPU's NUMA node:
    numactl --cpunodebind=0 --membind=0 python train.py  # GPU on node 0
""")

def get_available_cores() -> list:
    """
    TODO 2: Implement this function.
    Return a sorted list of CPU core IDs that this process is allowed
    to run on, using os.sched_getaffinity(0).

    If os.sched_getaffinity is not available (some OS configurations),
    fall back to list(range(os.cpu_count() or 1)).
    """
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


cores = get_available_cores()
assert cores is not None, "get_available_cores must return a list. Did you implement TODO 2?"
assert len(cores) >= 1, \
    f"get_available_cores must return at least 1 core, got {len(cores)}"
assert all(isinstance(c, int) for c in cores), \
    f"All elements must be int, got {[type(c) for c in cores]}"
assert cores == sorted(cores), \
    f"Cores must be in sorted order, got {cores}"

print(f"  Available CPU cores: {cores[:8]}{'...' if len(cores) > 8 else ''}")
print(f"  Total allowed cores: {len(cores)}")
print("  ✓ Section 2 passed — CPU affinity read successfully")


# ─────────────────────────────────────────────────────────────
# SECTION 3: NUMA Binding Commands
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: NUMA Binding Commands ──")
print("""
  numactl is the standard tool for binding a process to specific NUMA
  nodes.  The binding applies to BOTH CPU scheduling (which cores run
  the process) and memory allocation (which DRAM bank gets .malloc()).

  THE LINUX FIRST-TOUCH ALLOCATION POLICY:
    When you allocate memory (e.g., torch.rand(n)), Linux does NOT
    immediately assign physical pages.  Pages are assigned on first
    WRITE (first touch).  The NUMA node of the page is the node where
    the writing thread is running at that moment.
    → If your DataLoader worker allocates a buffer while running on
      socket 1 but needs to DMA it to socket 0's GPU, the data must
      travel across the interconnect.
    → numactl --membind=0 forces ALL allocations to socket 0 DRAM,
      regardless of which core does the first touch.

  PRACTICAL COMMANDS:

  Finding which NUMA node a GPU is on:
    for dev in /sys/bus/pci/devices/*/numa_node; do
        echo "$dev: $(cat $dev)"
    done

  Binding training to a specific NUMA node:
    numactl --cpunodebind=0 --membind=0 python train.py

  Multi-GPU, each GPU on a different node:
    CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 python train.py &
    CUDA_VISIBLE_DEVICES=1 numactl --cpunodebind=1 --membind=1 python train.py &

  Verify your binding is active (check the process's memory policy):
    cat /proc/$(pgrep -f train.py)/numa_maps | head -5

  Check NUMA hit/miss stats for a running process:
    numastat -p $(pgrep -f train.py)
    (look for numa_miss: high count = remote memory accesses)
""")

def build_numactl_command(numa_node: int, script: str) -> str:
    """
    TODO 3: Implement this function.
    Build the numactl command string to bind a Python script to a
    specific NUMA node for both CPU scheduling and memory allocation.
    Return:
      f"numactl --cpunodebind={numa_node} --membind={numa_node} python {script}"
    """
    return f"numactl --cpunodebind={numa_node} --membind={numa_node} python {script}"


cmd0 = build_numactl_command(0, "train.py")
cmd1 = build_numactl_command(1, "train.py")

assert cmd0 is not None, "build_numactl_command must return a string. Did you implement TODO 3?"
assert "cpunodebind=0" in cmd0, (
    f"numactl command for node 0 must contain 'cpunodebind=0', got: {cmd0}"
)
assert "membind=0" in cmd0, (
    f"numactl command for node 0 must contain 'membind=0', got: {cmd0}"
)
assert "cpunodebind=1" in cmd1, (
    f"numactl command for node 1 must contain 'cpunodebind=1', got: {cmd1}"
)
assert "membind=1" in cmd1, (
    f"numactl command for node 1 must contain 'membind=1', got: {cmd1}"
)
assert "train.py" in cmd0, \
    f"numactl command must include the script name, got: {cmd0}"

print(f"  build_numactl_command(0, 'train.py'):")
print(f"    {cmd0}")
print(f"  build_numactl_command(1, 'train.py'):")
print(f"    {cmd1}")
print("  ✓ Section 3 passed — numactl command builder correct")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Simulating NUMA Bandwidth Difference
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Simulating NUMA Bandwidth Difference ──")
print("""
  On a real multi-socket server you would measure remote memory access
  by running: numactl --cpunodebind=0 --membind=1 python measure.py
  and comparing with: numactl --cpunodebind=0 --membind=0 python measure.py

  The difference is the NUMA penalty — typically 10–30% lower bandwidth
  for the remote case, plus higher latency for small random accesses.

  Since we cannot force NUMA binding in a portable Python script,
  we SIMULATE the penalty by injecting a small delay per 64 KB chunk
  of data accessed.  This models the extra latency of crossing the
  UPI interconnect for each cache line that must be fetched from
  the remote socket's memory controller.

  ON A REAL SYSTEM, the penalty is continuous and hardware-imposed:
  every cache line that comes from the remote socket takes 1.5–2.5×
  longer to arrive than a local cache line.  Our simulation models
  this as a per-chunk overhead, which is a valid approximation for
  bandwidth-limited workloads.

  WHAT THE NUMBERS MEAN:
    local_bw  > 20 GB/s  → data in local DRAM, no interconnect overhead
    remote_bw = local_bw * (1 / (1 + penalty_fraction))
    Typical penalty_fraction: 0.20–0.50 (20–50% slower)
""")

def simulate_local_access(n: int) -> float:
    """
    TODO 4a: Implement this function.
    Create a float32 numpy array of n elements, compute its sum 5 times,
    and return the bandwidth in GB/s = (n * 4) / min_time / 1e9.
    Run 3 warmup iterations first.
    """
    arr = np.ones(n, dtype=np.float32)
    for _ in range(3):          # warmup
        _ = arr.sum()
    min_time = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        _ = arr.sum()
        min_time = min(min_time, time.perf_counter() - t0)
    return (n * 4) / min_time / 1e9


def simulate_remote_access(n: int, penalty_us: float) -> float:
    """
    TODO 4b: Implement this function.
    Simulate remote NUMA memory access by adding a time.sleep(penalty_us/1e6)
    delay for each 64 KB chunk of the array.

    Steps:
      1. Create float32 array of n elements
      2. chunk_size = 64 * 1024 // 4  (elements per 64 KB)
      3. n_chunks = max(1, n // chunk_size)
      4. Time 3 runs:
           total_sleep = n_chunks * penalty_us / 1e6
           time.sleep(total_sleep)    # simulate remote latency overhead
           _ = arr.sum()              # the actual memory access
      5. Return bandwidth = (n * 4) / min_time / 1e9
         where min_time includes both the sleep and the sum.
    """
    arr = np.ones(n, dtype=np.float32)
    chunk_size = 64 * 1024 // 4          # elements per 64 KB
    n_chunks = max(1, n // chunk_size)
    total_sleep = n_chunks * penalty_us / 1e6
    min_time = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        time.sleep(total_sleep)          # simulate remote latency overhead
        _ = arr.sum()                    # the actual memory access
        min_time = min(min_time, time.perf_counter() - t0)
    return (n * 4) / min_time / 1e9


local_bw = simulate_local_access(10_000_000)
remote_bw = simulate_remote_access(1_000_000, 10)   # 10 μs per 64 KB chunk

assert local_bw is not None and local_bw > 0, (
    "simulate_local_access(10_000_000) must return > 0 GB/s. "
    "Did you implement TODO 4a?"
)
assert remote_bw is not None and remote_bw > 0, (
    "simulate_remote_access(1_000_000, 10) must return > 0 GB/s. "
    "Did you implement TODO 4b?"
)

print(f"  simulate_local_access(10M elements):        {local_bw:.1f} GB/s")
print(f"  simulate_remote_access(1M elem, 10μs/64KB): {remote_bw:.1f} GB/s")

# Compute and show penalty at different simulated latency levels
print("\n  Simulated NUMA penalty at different per-chunk latencies:")
print(f"  {'Penalty (μs/64KB)':>20}  {'Simulated BW (GB/s)':>22}")
print(f"  {'─'*20}  {'─'*22}")
for penalty_us in [0, 5, 10, 20, 50]:
    bw = simulate_remote_access(500_000, penalty_us)
    print(f"  {penalty_us:>20}  {bw:>22.1f}")

print("  ✓ Section 4 passed — local and remote access simulation returns > 0 GB/s")


print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 9.2 complete!")
print()
print("  You can now parse numactl output, read CPU affinity,")
print("  build numactl binding commands, and simulate the NUMA")
print("  remote memory penalty.")
print()
print("  Next: III.Linux_Systems_Profiling/9.Memory_Hierarchy_and_NUMA/9.3_memory_bandwidth.py")
print("=" * 60)
