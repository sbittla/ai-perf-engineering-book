# Chapter 8 — perf, eBPF, and Flamegraphs

Exercises for Chapter 8 of *AI Systems Performance Engineering*.

## Exercise map

| File | Sections | Topics | Est. time |
|---|---|---|---|
| `8.1_perf_fundamentals.py` | 4 | IPC measurement, sequential vs random cache miss ratio, branch misprediction timing, perf stat command reference and `interpret_ipc()` | 45 min |
| `8.2_cpu_flamegraphs.py` | 4 | Reading flamegraph layout, hotspot identification, cProfile on a mixed workload, differential profiling with `profile_fn()`, frame width classification | 45 min |
| `8.3_ebpf_and_bpftrace.py` | 4 | eBPF verifier and safety model, five key AI engineering tools, opensnoop file access simulation, biolatency histogram builder, bpftrace one-liner reference | 45 min |
| `8.4_lock_contention.py` | 4 | Python GIL timing for CPU-bound tasks, lock-per-increment shared counter, ProcessPoolExecutor worker tasks, context-switch rate contention indicator | 45 min |

## Running

```bash
python "III.Linux_Systems_Profiling/8.Perf_eBPF_and_Flamegraphs/8.1_perf_fundamentals.py"
python "III.Linux_Systems_Profiling/8.Perf_eBPF_and_Flamegraphs/8.2_cpu_flamegraphs.py"
python "III.Linux_Systems_Profiling/8.Perf_eBPF_and_Flamegraphs/8.3_ebpf_and_bpftrace.py"
python "III.Linux_Systems_Profiling/8.Perf_eBPF_and_Flamegraphs/8.4_lock_contention.py"
```

## Hardware and tool requirements

- **No GPU required.** All exercises run on CPU.
- **No sudo required.** The exercises print perf, bpftrace, and strace commands for optional use in a side terminal, but the exercises themselves run without any of those tools installed.
- **perf stat (optional):** Install with `sudo apt install linux-perf`. Run the printed commands alongside the exercises for real hardware counter data.
- **py-spy (optional):** Install with `pip install py-spy`. Provides Python-level flamegraphs; mentioned in 8.2 but not required.
- **bpftrace (optional):** Install with `sudo apt install bpftrace`. The one-liners in 8.3 require root when run.

## Side-terminal commands for deeper measurement

While running `8.1_perf_fundamentals.py`, you can measure real hardware counters:

```bash
# IPC, cache misses, branch mispredictions — all five core counters:
sudo perf stat -e cache-misses,cache-references,branch-misses,instructions,cycles \
    python "III.Linux_Systems_Profiling/8.Perf_eBPF_and_Flamegraphs/8.1_perf_fundamentals.py"

# Detailed cache hierarchy — how many misses at each level:
sudo perf stat -e L1-dcache-misses,L2-dcache-misses,LLC-misses \
    python "III.Linux_Systems_Profiling/8.Perf_eBPF_and_Flamegraphs/8.1_perf_fundamentals.py"
```

While running `8.4_lock_contention.py`, you can measure real context switches and futex calls:

```bash
# Count system calls including futex (locks):
strace -c python "III.Linux_Systems_Profiling/8.Perf_eBPF_and_Flamegraphs/8.4_lock_contention.py"

# Watch context switches per second:
vmstat 1 10
```
