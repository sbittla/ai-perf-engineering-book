# Part VI — Capstone Projects

Four self-contained portfolio projects that integrate the tools and techniques from Parts I–V into complete, end-to-end performance engineering workflows. Each capstone follows the same structure: establish a baseline, profile to find the bottleneck, apply fixes one at a time, and document the before/after comparison with measured speedups.

## Chapter Map

| Chapter | Project | Theme | Exercises |
|---------|---------|-------|-----------|
| 16 — LLM Inference Optimization Lab | Optimize a mini GPT to 2–3× baseline | Production readiness, precision, torch.compile, profiling | 16.1, 16.2, 16.3, 16.4 |
| 17 — DataLoader Bottleneck Hunt | Find and fix 3 DataLoader bugs | CPU-GPU starvation diagnosis | 17.1, 17.2 |
| 18 — KV Cache Memory Pressure | Plan LLM deployment given VRAM budget | KV cache formula, OOM threshold | 18.1, 18.2 |
| 19 — Flamegraph Challenge | Find 3 training loop bugs via profiling | .item() sync, H2D, no_grad | 19.1, 19.2 |

## Quick Start

```bash
# Capstone 16: LLM Inference Optimization
python 16.LLM_Inference_Optimization/16.1_production_readiness.py
python 16.LLM_Inference_Optimization/16.2_baseline_inference.py
python 16.LLM_Inference_Optimization/16.3_precision_and_compile.py
python 16.LLM_Inference_Optimization/16.4_profiling_audit.py

# Capstone 17: DataLoader Bottleneck Hunt
python 17.DataLoader_Bottleneck_Hunt/17.1_slow_dataloader.py
python 17.DataLoader_Bottleneck_Hunt/17.2_fast_dataloader.py

# Capstone 18: KV Cache Memory Pressure
python 18.KV_Cache_Memory_Pressure/18.1_kv_cache_scaling.py
python 18.KV_Cache_Memory_Pressure/18.2_memory_budget_planning.py

# Capstone 19: Flamegraph Challenge
python 19.Flamegraph_Challenge/19.1_slow_training_analysis.py
python 19.Flamegraph_Challenge/19.2_optimized_training.py
```

## The Capstone Methodology

Every capstone follows the same structured workflow:

```
1. BASELINE    → measure TTFT/TPS/idle_pct/step_ms before any changes
2. PROFILE     → find the bottleneck (profiler, idle_pct, flamegraph)
3. ISOLATE     → fix ONE thing; measure; document speedup
4. COMBINE     → apply all fixes; final comparison table
5. REPORT      → save JSON with before/after metrics
```

This is the standard of evidence required for production performance work: every speedup is attributed to a specific change, measured in isolation, with reproducible numbers.

## Cross-Capstone Skills

| Skill | Capstone(s) |
|-------|------------|
| Arithmetic intensity and roofline | 16.2 (FP16 vs FP32 classification) |
| torch.profiler + record_function | 16.3 |
| GPU idle percentage measurement | 17.1, 17.2 |
| KV cache formula | 18.1, 18.2 |
| .item() sync trap | 19.1, 19.2 |
| non_blocking H2D transfer | 17.2, 19.2 |
| torch.no_grad() in evaluation | 19.2 |
| Memory budget decomposition | 18.2 |

## Prerequisites

All capstones require Parts I–V. CPU-only fallbacks are included throughout — GPU is not required to run the exercises, though CUDA-specific measurements (SM%, BW%) will show representative numbers only on GPU.

## Exercise Index

| File | What You Build |
|------|---------------|
| `16.1_production_readiness.py` | Research-vs-production gap; six-dimension readiness self-assessment (read-and-run background, no GPU) |
| `16.2_baseline_inference.py` | MiniGPT baseline: TTFT, TPS, VRAM snapshot |
| `16.3_precision_and_compile.py` | FP16 + compile optimization ladder with speedup table |
| `16.4_profiling_audit.py` | Per-op time attribution; diagnosis-to-fix report |
| `17.1_slow_dataloader.py` | Three-bug DataLoader; idle_pct measurement |
| `17.2_fast_dataloader.py` | Fix all three bugs; 43× improvement on I/O-heavy load |
| `18.1_kv_cache_scaling.py` | OOM threshold, latency model, strategy comparison |
| `18.2_memory_budget_planning.py` | Deployment plan for chat/RAG/batch use cases |
| `19.1_slow_training_analysis.py` | Three training loop bugs; cost per bug |
| `19.2_optimized_training.py` | All fixes applied; flamegraph interpretation |
