# Chapter 4 Exercises — The CUDA Execution Model

Three exercises covering the GPU's programming model and memory hierarchy from the perspective of a performance engineer.

| File | Section | Topics | Estimated Time |
|---|---|---|---|
| 4.1_execution_model.py | 4.1 | Thread hierarchy, SIMT, SM occupancy, warp divergence | 45 min |
| 4.2_memory_coalescing.py | 4.2 | Stride-1 vs strided access, cache tiers, contiguous() | 45 min |
| 4.3_tensor_cores_and_fusion.py | 4.3 | Tensor Cores, alignment, BF16, torch.compile fusion | 45 min |

## Hardware note

All three exercises require a CUDA GPU for the benchmarking sections to produce meaningful results. On a CPU-only machine, the timing sections are skipped and reference values are printed, but the conceptual assertions still run.

Minimum recommended GPU: RTX 3060 (8 GB). An A100 or H100 will show the largest Tensor Core speedups.

Complete all three before moving to Chapter 5.
