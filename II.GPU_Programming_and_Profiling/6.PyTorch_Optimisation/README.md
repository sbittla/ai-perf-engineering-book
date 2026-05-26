# Chapter 6 Exercises — PyTorch Optimisation

Three exercises covering the three highest-leverage PyTorch optimisation techniques: precision, quantization, and compilation.

| File | Section | Topics | Estimated Time |
|---|---|---|---|
| 6.1_precision_and_amp.py | 6.1 | FP32/FP16/BF16 memory cost, autocast, GradScaler training | 60 min |
| 6.2_quantization.py | 6.2 | INT8 dynamic quantization, memory footprint, INT4 concepts | 45 min |
| 6.3_torch_compile.py | 6.3 | torch.compile modes, warmup, mode comparison, batch sweep | 60 min |

## Hardware note

**6.1** (AMP): Requires a CUDA GPU for the autocast memory savings and GradScaler sections. CPU fallback is provided.

**6.2** (quantization): Runs entirely on CPU. `torch.quantization.quantize_dynamic` is CPU-only by design. No GPU required.

**6.3** (torch.compile): Runs on both CPU and CUDA. Compilation is faster on GPU; on CPU, torch.compile may not show speedups for the models used here but the API behaviour is the same.

Complete all three before moving to Chapter 7.
