# Chapter 5 Exercises — GPU Profiling

Three exercises covering the three-tier profiling hierarchy: system timeline (nsys), kernel hardware counters (ncu), and PyTorch operator profiling (torch.profiler).

| File | Section | Topics | Estimated Time |
|---|---|---|---|
| 5.1_nsys_profiling.py | 5.1 | NVTX markers, CPU-GPU overlap, idle_fraction, training step annotation | 60 min |
| 5.2_ncu_profiling.py | 5.2 | Arithmetic intensity, roofline classification, ncu metrics reference | 45 min |
| 5.3_torch_profiler.py | 5.3 | Profiler schedule, Chrome trace export, Self CUDA analysis | 45 min |

## Hardware note

5.1 and 5.2 are designed to be run under the respective profiling tools:

```bash
# 5.1: annotate with nsys
nsys profile --trace=cuda,nvtx \
    -o /tmp/5.1_output \
    python "II.GPU_Programming_and_Profiling/5.GPU_Profiling/5.1_nsys_profiling.py"

# 5.2: profile kernels with ncu
ncu --set basic --kernel-name gemm \
    -o /tmp/5.2_output \
    python "II.GPU_Programming_and_Profiling/5.GPU_Profiling/5.2_ncu_profiling.py"
```

nsys and ncu must be installed from the NVIDIA Nsight suite:
- Nsight Systems: https://developer.nvidia.com/nsight-systems
- Nsight Compute: https://developer.nvidia.com/nsight-compute

5.3 runs standalone and exports a Chrome trace to `/tmp/pt2_chrome_trace.json`.
Open it in Chrome at `chrome://tracing` or at https://ui.perfetto.dev.

Complete all three before moving to Chapter 6.
