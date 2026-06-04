# Part IX — Production Capstones

The closing capstones tie the whole book together on production-scale problems: serving an LLM under load, scaling across multiple GPUs, and optimizing cloud cost. All exercises run on CPU-only machines using analytic models and reference data — no GPU, cluster, or cloud account is required.

## Chapters

| Chapter | Topic | Exercises |
|---------|-------|-----------|
| 28 | Capstone 5 — Production LLM Serving Benchmark (vLLM) | `28.1_serving_benchmark.py` |
| 29 | Capstone 6 — Multi-GPU Scaling Challenge | `29.1_scaling_challenge.py` |
| 30 | Capstone 7 — Cloud Cost Optimization | `30.1_cost_optimization.py` |

## Run Order

```bash
python IX.Production_Capstones/28.Production_Serving/28.1_serving_benchmark.py
python IX.Production_Capstones/29.MultiGPU_Scaling/29.1_scaling_challenge.py
python IX.Production_Capstones/30.Cloud_Cost/30.1_cost_optimization.py
```

## Prerequisites

- Python 3.10+
- `torch` (optional — exercises degrade gracefully to reference data without it)
- No GPU, cluster, or cloud credentials required

> **What these produce:** each capstone emits a before/after comparison and the headline
> metric a hiring manager looks for — throughput/latency under load and cost-per-token (Ch 28),
> scaling efficiency vs the Amdahl ceiling (Ch 29), and the accelerator cost matrix (Ch 30).
