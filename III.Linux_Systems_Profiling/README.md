# Part III — Linux Systems Profiling

Companion exercises for **Part III — Linux Systems Profiling** of *AI Systems Performance Engineering*.

Each folder name matches a chapter title exactly as it appears in the book.

## Structure

```
III.Linux_Systems_Profiling/
├── 8.Perf_eBPF_and_Flamegraphs/
│   ├── 8.1_perf_fundamentals.py
│   ├── 8.2_cpu_flamegraphs.py
│   ├── 8.3_ebpf_and_bpftrace.py
│   └── 8.4_lock_contention.py
└── 9.Memory_Hierarchy_and_NUMA/
    ├── 9.1_memory_hierarchy.py
    ├── 9.2_numa_and_topology.py
    └── 9.3_memory_bandwidth.py
```

## Setup

```bash
git clone https://github.com/your-handle/ai-perf-engineering-book
cd ai-perf-engineering-book

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install torch numpy
```

All Part III exercises run on CPU (no GPU required). NumPy and PyTorch must be installed.

## Running the exercises (in order)

```bash
# Chapter 8 — perf, eBPF, and Flamegraphs
python "III.Linux_Systems_Profiling/8.Perf_eBPF_and_Flamegraphs/8.1_perf_fundamentals.py"
python "III.Linux_Systems_Profiling/8.Perf_eBPF_and_Flamegraphs/8.2_cpu_flamegraphs.py"
python "III.Linux_Systems_Profiling/8.Perf_eBPF_and_Flamegraphs/8.3_ebpf_and_bpftrace.py"
python "III.Linux_Systems_Profiling/8.Perf_eBPF_and_Flamegraphs/8.4_lock_contention.py"

# Chapter 9 — Memory Hierarchy and NUMA
python "III.Linux_Systems_Profiling/9.Memory_Hierarchy_and_NUMA/9.1_memory_hierarchy.py"
python "III.Linux_Systems_Profiling/9.Memory_Hierarchy_and_NUMA/9.2_numa_and_topology.py"
python "III.Linux_Systems_Profiling/9.Memory_Hierarchy_and_NUMA/9.3_memory_bandwidth.py"
```

Each file prints `✓` for every passing section and stops with a clear assertion error at the first incomplete TODO.

## Exercise map — book section to file

| Chapter | Topic | File |
|---|---|---|
| 8.1 | IPC, cache miss rate, branch misprediction, perf stat events | `8.Perf_eBPF_and_Flamegraphs/8.1_perf_fundamentals.py` |
| 8.2 | Flamegraph structure, hotspot identification, py-spy, cProfile, differential profiling | `8.Perf_eBPF_and_Flamegraphs/8.2_cpu_flamegraphs.py` |
| 8.3 | eBPF safety, opensnoop, biolatency, bpftrace one-liners, I/O latency categories | `8.Perf_eBPF_and_Flamegraphs/8.3_ebpf_and_bpftrace.py` |
| 8.4 | Python GIL, shared counter contention, multiprocessing vs threading, vmstat cs column | `8.Perf_eBPF_and_Flamegraphs/8.4_lock_contention.py` |
| 9.1 | Cache line size, stride bandwidth, working set tiers, matrix layout, .contiguous() | `9.Memory_Hierarchy_and_NUMA/9.1_memory_hierarchy.py` |
| 9.2 | NUMA nodes, CPU affinity, numactl binding, remote memory simulation | `9.Memory_Hierarchy_and_NUMA/9.2_numa_and_topology.py` |
| 9.3 | CPU DRAM bandwidth, PCIe H2D bandwidth, bandwidth hierarchy, roofline ridge point | `9.Memory_Hierarchy_and_NUMA/9.3_memory_bandwidth.py` |

## Hardware and tool requirements

- **CPU-only:** All 7 exercises run entirely on CPU. No GPU is required.
- **CUDA GPU (optional):** Exercise 9.3 section 2 measures PCIe bandwidth if CUDA is available. All other sections run on CPU with the same results.
- **perf (optional):** Chapter 8 exercises print the exact `perf stat` commands to run in a side terminal. The exercises themselves work without `perf` installed — they use Python timing as a proxy.
- **bpftrace / bpfcc-tools (optional):** Chapter 8.3 prints bpftrace one-liners for reference. The exercises run without bpftrace installed.
- **numactl (optional):** Chapter 9.2 calls `numactl --hardware` if available and falls back gracefully if not installed.
- **sudo:** Not required for any exercise. The perf and bpftrace commands shown require sudo when run externally, but the exercises themselves do not.
