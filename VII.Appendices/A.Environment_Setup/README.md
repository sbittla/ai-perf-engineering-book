# Appendix A — Environment Setup

Verify every component required by the book before starting.

## Exercises

| File | Topic | Key Sections |
|------|-------|-------------|
| `A.1_environment_check.py` | Full environment verification | Python, PyTorch, GPU, packages, tools, timing baseline |

## Workflow

```bash
python A.1_environment_check.py
```

## What It Checks

| Section | Check | Minimum |
|---------|-------|---------|
| 1 — Python & PyTorch | Version | Python 3.10+, PyTorch 2.0+ |
| 2 — GPU Capabilities | VRAM, compute capability, Tensor Cores | Any CUDA-capable GPU |
| 3 — Python Packages | Required + optional packages | torch, numpy, matplotlib, tqdm |
| 4 — Profiling Tools | nsys, ncu, py-spy, perf, flamegraph | All optional but recommended |
| 5 — Timing Baseline | CUDA events vs wall-clock | Confirms CUDA events work |

CPU-only machines pass all sections (GPU-specific checks print a notice and skip).

## Full Setup Guide

See `docs/HARDWARE_SETUP.md` for step-by-step driver, CUDA, and package installation.

## RTX 4060 Quick Reference

```
Architecture : Ada Lovelace (CC 8.9)
VRAM         : 8 GB GDDR6
Bandwidth    : ~272 GB/s
Peak FP16    : ~136 TFLOP/s (Tensor Cores)
Tensor Cores : Yes (4th gen — FP16, BF16, INT8, FP8)
NVLink       : No (consumer GPU)
PCIe         : 4.0 x8
```

LLM feasibility on 8 GB:
- GPT-2 (124M) FP16: ✓
- Llama-7B INT4 (GPTQ/AWQ): ✓ ~3.5 GB
- Llama-7B INT8: ✓ ~7.0 GB
- Llama-7B FP16: ✗ needs 14 GB
