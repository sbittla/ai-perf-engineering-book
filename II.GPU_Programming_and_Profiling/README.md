# Part II — GPU Programming & Profiling

Companion exercises for **Part II — GPU Programming & Profiling** of *AI Performance Engineering*.

Each folder name matches a chapter title exactly as it appears in the book.

## Structure

```
II.GPU_Programming_and_Profiling/
├── 4.The_CUDA_Execution_Model/
│   ├── 4.1_cuda_foundations.py
│   ├── 4.2_execution_model.py
│   ├── 4.3_memory_coalescing.py
│   └── 4.4_tensor_cores_and_fusion.py
├── 5.GPU_Profiling/
│   ├── 5.1_nsys_profiling.py
│   ├── 5.2_ncu_profiling.py
│   └── 5.3_torch_profiler.py
├── 6.PyTorch_Optimization/
│   ├── 6.1_precision_and_amp.py
│   ├── 6.2_quantization.py
│   └── 6.3_torch_compile.py
└── 7.DataLoader_Optimization/
    ├── 7.1_dataloader_pipeline.py
    └── 7.2_io_bottleneck.py
```

## Setup

```bash
git clone https://github.com/your-handle/ai-perf-engineering-book
cd ai-perf-engineering-book

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

A CUDA-capable GPU is required for Chapters 4–7. Most exercises skip CUDA sections gracefully on CPU-only machines, but the profiling exercises (5.1, 5.2) require a real GPU to be meaningful.

## Running the exercises (in order)

```bash
# Chapter 4 — The CUDA Execution Model
python "II.GPU_Programming_and_Profiling/4.The_CUDA_Execution_Model/4.1_cuda_foundations.py"
python "II.GPU_Programming_and_Profiling/4.The_CUDA_Execution_Model/4.2_execution_model.py"
python "II.GPU_Programming_and_Profiling/4.The_CUDA_Execution_Model/4.3_memory_coalescing.py"
python "II.GPU_Programming_and_Profiling/4.The_CUDA_Execution_Model/4.4_tensor_cores_and_fusion.py"

# Chapter 5 — GPU Profiling
python "II.GPU_Programming_and_Profiling/5.GPU_Profiling/5.1_nsys_profiling.py"
python "II.GPU_Programming_and_Profiling/5.GPU_Profiling/5.2_ncu_profiling.py"
python "II.GPU_Programming_and_Profiling/5.GPU_Profiling/5.3_torch_profiler.py"

# Chapter 6 — PyTorch Optimization
python "II.GPU_Programming_and_Profiling/6.PyTorch_Optimization/6.1_precision_and_amp.py"
python "II.GPU_Programming_and_Profiling/6.PyTorch_Optimization/6.2_quantization.py"
python "II.GPU_Programming_and_Profiling/6.PyTorch_Optimization/6.3_torch_compile.py"

# Chapter 7 — DataLoader Optimization
python "II.GPU_Programming_and_Profiling/7.DataLoader_Optimization/7.1_dataloader_pipeline.py"
python "II.GPU_Programming_and_Profiling/7.DataLoader_Optimization/7.2_io_bottleneck.py"
```

Each file prints `✓` for every passing section and stops with a clear error at the first incomplete TODO.

## Exercise map — book section to file

| Chapter | Topic | File |
|---|---|---|
| II.1 | The CUDA story, GPU/SM/warp/shared-memory detection, CUDA stack overview (Part II background, read-and-run) | `4.The_CUDA_Execution_Model/4.1_cuda_foundations.py` |
| 4.1 | GPU thread hierarchy, SIMT, occupancy, warp divergence | `4.The_CUDA_Execution_Model/4.2_execution_model.py` |
| 4.2 | Memory coalescing, cache tiers, row/column access | `4.The_CUDA_Execution_Model/4.3_memory_coalescing.py` |
| 4.3 | Tensor Cores, alignment, BF16, kernel fusion | `4.The_CUDA_Execution_Model/4.4_tensor_cores_and_fusion.py` |
| 5.1 | NVTX markers, CPU-GPU overlap, idle_fraction | `5.GPU_Profiling/5.1_nsys_profiling.py` |
| 5.2 | Arithmetic intensity, roofline, ncu metrics | `5.GPU_Profiling/5.2_ncu_profiling.py` |
| 5.3 | torch.profiler schedule, Chrome trace, Self CUDA | `5.GPU_Profiling/5.3_torch_profiler.py` |
| 6.1 | FP32/FP16/BF16 memory, autocast, GradScaler | `6.PyTorch_Optimization/6.1_precision_and_amp.py` |
| 6.2 | INT8 dynamic quantization, memory footprint, INT4 | `6.PyTorch_Optimization/6.2_quantization.py` |
| 6.3 | torch.compile modes, warmup, batch size sweep | `6.PyTorch_Optimization/6.3_torch_compile.py` |
| 7.1 | num_workers, pin_memory, prefetch_factor, .item() | `7.DataLoader_Optimization/7.1_dataloader_pipeline.py` |
| 7.2 | GPU starvation, idle_pct, nvidia-smi diagnosis, fix checklist | `7.DataLoader_Optimization/7.2_io_bottleneck.py` |

## Hardware requirements

- **CPU-only:** All exercises run and skip CUDA sections gracefully.
- **CUDA GPU (minimum RTX 3060 / 8 GB):** Required for meaningful results in Chapters 4–7. CUDA sections are silently skipped without a GPU but the concepts still apply.
- **Profiling tools:** nsys and ncu (from NVIDIA's Nsight suite) are needed for Chapters 5.1 and 5.2 respectively. Install from [developer.nvidia.com/nsight-systems](https://developer.nvidia.com/nsight-systems).
