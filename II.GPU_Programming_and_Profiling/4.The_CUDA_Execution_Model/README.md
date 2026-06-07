# Chapter 4 Exercises — The CUDA Execution Model

Four exercises covering the GPU's programming model and memory hierarchy from the perspective of a performance engineer.

| File | Section | Topics | Estimated Time |
|---|---|---|---|
| 4.1_cuda_foundations.py | II.1 | The CUDA story, GPU/SM/warp/shared-memory detection, CUDA stack overview (Part II background, read-and-run) | 20 min |
| 4.2_execution_model.py | 4.1 | Thread hierarchy, SIMT, SM occupancy, warp divergence | 45 min |
| 4.3_memory_coalescing.py | 4.2 | Stride-1 vs strided access, cache tiers, contiguous() | 45 min |
| 4.4_tensor_cores_and_fusion.py | 4.3 | Tensor Cores, alignment, BF16, torch.compile fusion | 45 min |

## Hardware note

The benchmarking exercises (4.2–4.4) require a CUDA GPU for the timing sections to produce meaningful results; `4.1_cuda_foundations.py` is a read-and-run background survey. On a CPU-only machine, the timing sections are skipped and reference values are printed, but the conceptual assertions still run.

Minimum recommended GPU: RTX 3060 (8 GB). An A100 or H100 will show the largest Tensor Core speedups.

Complete all four before moving to Chapter 5.
