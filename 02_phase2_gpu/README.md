# Phase 2 — GPU Programming & CUDA Profiling

**Duration: 5 weeks — THE most critical phase of this course.**

## Module 3: CUDA Fundamentals
```bash
python module3_cuda_fundamentals/vector_add.py           # threads, blocks, streams
python module3_cuda_fundamentals/occupancy_experiment.py # occupancy deep dive
python module3_cuda_fundamentals/matmul_bench.py         # roofline model
python module3_cuda_fundamentals/cuda_kernels.py         # 6 GPU experiments
```

## Module 4: GPU Profiling
```bash
# Profile training with all tools
nsys profile --stats=true python module4_profiling/train.py --steps 50
ncu --set basic python module4_profiling/train.py --steps 5
python module4_profiling/profile_pytorch_infer.py
python module4_profiling/optimize_batch_size.py
```

## Module 5: PyTorch Performance Engineering
```bash
python module5_pytorch_perf/fp16_bf16_bench.py
python module5_pytorch_perf/quantization_bench.py
python module5_pytorch_perf/llama_infer_optimize.py   # full optimisation ladder
```

## Key Learning Outcomes
- Understand the GPU execution model (warps, blocks, grids, occupancy)
- Read and interpret nsys timelines and ncu roofline plots
- Know when a kernel is memory-bound vs compute-bound
- Apply FP16 and torch.compile to any PyTorch model
