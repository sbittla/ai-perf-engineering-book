# Chapter 9 — Memory Hierarchy and NUMA

Exercises for Chapter 9 of *AI Performance Engineering*.

## Exercise map

| File | Sections | Topics | Est. time |
|---|---|---|---|
| `9.1_memory_hierarchy.py` | 4 | Stride bandwidth (1/4/8/16/32), working set size and cache tier classification, row vs column matrix access, PyTorch `.contiguous()` and `is_contiguous()` | 45 min |
| `9.2_numa_and_topology.py` | 4 | Parsing `numactl --hardware` output, CPU affinity with `os.sched_getaffinity`, numactl binding command builder, local vs remote memory simulation | 45 min |
| `9.3_memory_bandwidth.py` | 4 | CPU DRAM bandwidth via `torch.Tensor.clone()`, PCIe H2D bandwidth (pinned vs pageable), full bandwidth hierarchy table, roofline ridge point with measured bandwidth | 45 min |

## Running

```bash
python "III.Linux_Systems_Profiling/9.Memory_Hierarchy_and_NUMA/9.1_memory_hierarchy.py"
python "III.Linux_Systems_Profiling/9.Memory_Hierarchy_and_NUMA/9.2_numa_and_topology.py"
python "III.Linux_Systems_Profiling/9.Memory_Hierarchy_and_NUMA/9.3_memory_bandwidth.py"
```

## Hardware and tool requirements

- **No GPU required for 9.1 and 9.2.** Exercise 9.3 section 2 measures PCIe bandwidth if CUDA is available; otherwise it prints a note and skips that measurement.
- **numactl (optional):** Exercise 9.2 calls `numactl --hardware` if installed; falls back gracefully otherwise. Install with `sudo apt install numactl`.
- **Large memory recommended:** Exercises 9.1 and 9.3 allocate 256 MB arrays to overflow the L3 cache. A system with at least 2 GB of free RAM is sufficient.

## Understanding your hardware numbers

To find your CPU's cache sizes:

```bash
# L3 cache size:
cat /sys/devices/system/cpu/cpu0/cache/index3/size

# Full cache hierarchy:
lscpu | grep -i cache

# Full topology including NUMA:
numactl --hardware       # requires numactl package
lstopo --of ascii        # requires hwloc package
```

To find which NUMA node a GPU is attached to:

```bash
for dev in /sys/bus/pci/devices/*/numa_node; do
    echo "$dev: $(cat $dev)"
done
```

## Side-terminal commands for deeper measurement

While running `9.1_memory_hierarchy.py`, measure real cache miss rates:

```bash
sudo perf stat -e cache-misses,cache-references,LLC-misses \
    python "III.Linux_Systems_Profiling/9.Memory_Hierarchy_and_NUMA/9.1_memory_hierarchy.py"
```

To measure NUMA effects directly (requires a multi-socket server and numactl):

```bash
# Local memory (fast):
numactl --cpunodebind=0 --membind=0 \
    python "III.Linux_Systems_Profiling/9.Memory_Hierarchy_and_NUMA/9.3_memory_bandwidth.py"

# Remote memory (slower by 10–30%):
numactl --cpunodebind=0 --membind=1 \
    python "III.Linux_Systems_Profiling/9.Memory_Hierarchy_and_NUMA/9.3_memory_bandwidth.py"
```
