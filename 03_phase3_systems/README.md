# Phase 3 — Systems Profiling & Kernel-Level Analysis

**Duration: 4 weeks**

## Module 6: Linux Low-Level Profiling
```bash
python module6_linux_profiling/lock_contention.py
strace -c python module6_linux_profiling/lock_contention.py
python module6_linux_profiling/cache_miss_analysis.py
sudo perf stat -e cache-misses python module6_linux_profiling/cache_miss_analysis.py
```

## Module 7: Memory & NUMA
```bash
python module7_memory_numa/memory_bench.py
python module7_memory_numa/numa_workload.py --label default
numactl --cpunodebind=0 --membind=0 python module7_memory_numa/numa_workload.py --label node0
```
