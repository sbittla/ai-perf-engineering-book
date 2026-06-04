# Part VIII — Advanced Performance Engineering

This part covers advanced techniques used in production AI systems: writing custom GPU kernels, IO-aware attention, distributed-training scaling, and production observability. All exercises run on CPU-only machines using analytic models and reference data — no Triton, GPU, cluster, or monitoring stack is required.

## Chapters

| Chapter | Topic | Exercises |
|---------|-------|-----------|
| 24 | Triton — Writing Custom GPU Kernels | `24.1_triton_kernels.py` |
| 25 | FlashAttention — IO-Aware Attention | `25.1_flash_attention.py` |
| 26 | Distributed Training Performance | `26.1_scaling_analysis.py` |
| 27 | Production Observability | `27.1_observability_metrics.py` |

## Run Order

```bash
python VIII.Advanced_Performance/24.Triton/24.1_triton_kernels.py
python VIII.Advanced_Performance/25.FlashAttention/25.1_flash_attention.py
python VIII.Advanced_Performance/26.Distributed_Training/26.1_scaling_analysis.py
python VIII.Advanced_Performance/27.Observability/27.1_observability_metrics.py
```

## Prerequisites

- Python 3.10+
- `torch` (optional — exercises degrade gracefully to reference data without it)
- No GPU, Triton, or cluster required

> **Go deeper:** hardware-based versions of these topics — a real Triton kernel, a vLLM
> internals trace, multi-node NCCL tuning, etc. — are specified as advanced capstones in
> [`Appendices/E.Advanced_Capstones/`](../Appendices/E.Advanced_Capstones) (book Appendix E).
