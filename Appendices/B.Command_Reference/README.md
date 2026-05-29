# Appendix B — Command Reference

Every profiling command used in the book, in one place.

## Exercises

| File | Topic | Key Sections |
|------|-------|-------------|
| `B.1_profiling_cheatsheet.py` | Diagnostic decision tree, profiler demo, command tables | nsys, ncu, torch.profiler, py-spy, perf, eBPF, nvidia-smi |

## Workflow

```bash
# Run the cheatsheet and demo
python B.1_profiling_cheatsheet.py
```

## Diagnostic Decision Tree

| Symptom | First Tool |
|---------|-----------|
| Is the GPU busy at all? | `nvidia-smi` (live util %) |
| Where is time going? | `nsys profile --stats=true` |
| Which kernel is the hotspot? | `torch.profiler` (PyTorch ops) |
| Is this kernel memory-bound? | `ncu --metrics dram__throughput` |
| Why is the CPU slow? | `py-spy record` (CPU flamegraph) |
| Is disk I/O starving the GPU? | `iostat -xz 1 && vmstat 1` |
| Is there NUMA penalty? | `numastat && numactl --hardware` |
| Is NCCL the bottleneck? | `nsys --trace=cuda,nvtx,nccl` |

## Tool Quick Reference

```bash
# GPU timeline
nsys profile --stats=true python train.py

# Per-kernel hardware counters
ncu --set basic python matmul_bench.py

# PyTorch op attribution
python -m torch.utils.bottleneck script.py

# CPU flamegraph
py-spy record -o flame.svg -- python train.py

# Live GPU dashboard
nvitop -m full

# Linux hardware events
perf stat -e cycles,instructions,cache-misses python train.py
```

## Full Reference

See `docs/COMMANDS.md` for the complete command reference with all flags and options.
