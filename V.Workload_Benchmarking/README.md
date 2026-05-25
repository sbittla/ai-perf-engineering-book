# Part V — Workload Benchmarking

This part teaches you to measure before optimising: how to avoid benchmarking mistakes, characterise any workload rigorously, port it from CPU to GPU correctly, and keep the DataLoader from starving the GPU.

## Chapter Map

| Chapter | Topic | Exercises |
|---------|-------|-----------|
| 14 — Benchmarking Methodology | Five common mistakes, T-L curves, 8-point characterisation | 14.1, 14.2, 14.3 |
| 15 — Porting a Workload | 6-step porting checklist, bottleneck shift, DataLoader tuning | 15.1, 15.2, 15.3 |

## Quick Start

```bash
# Chapter 14 — Benchmarking Methodology
python 14.Benchmarking_Methodology/14.1_benchmark_mistakes.py
python 14.Benchmarking_Methodology/14.2_throughput_latency_curve.py
python 14.Benchmarking_Methodology/14.3_workload_characterization.py

# Chapter 15 — Porting a Workload
python 15.Porting_a_Workload/15.1_porting_checklist.py
python 15.Porting_a_Workload/15.2_bottleneck_shift.py
python 15.Porting_a_Workload/15.3_dataloader_at_scale.py
```

## Key Ideas

### Benchmark Correctly First (Chapter 14)

Never report performance without: warmup, CUDA events (not `time.time()`), ≥ 50 samples, P99, and sustained throughput (not burst). These mistakes routinely produce 2–10× inflated numbers.

### Characterise Before Optimising (Chapter 14)

Run the 8-point checklist before every optimisation round:
- DRAM%, SM%, roofline position (arithmetic intensity vs ridge point)
- P50/P99 latency, throughput, GPU memory footprint
- CPU utilisation, thermal throttling

Save results as JSON. Diff reports before and after each optimisation round.

### Port Correctly, Then Optimise (Chapter 15)

Common porting mistakes are hard to diagnose after the fact:
- NumPy float64 → GPU float32 (`safe_from_numpy()`)
- Non-contiguous tensors (`.T` needs `.contiguous()` before CUDA kernels)
- Too many H2D copies in the hot path
- Not validating output correctness before benchmarking

### Anticipate Bottleneck Shift (Chapter 15)

Each optimisation exposes the next bottleneck:
```
CPU FP32 → [50× speedup] → GPU FP32 → [2-5×] → GPU FP16 → [1.5-2×] → compile
```
Measure throughput scaling with batch size to diagnose which rung you're on.

### Keep the GPU Fed (Chapter 15)

After GPU optimisation, the DataLoader is often the new bottleneck.
Measure `idle_pct`. If > 10%, tune: `num_workers`, `pin_memory=True`, `prefetch_factor`, `persistent_workers=True`.

## Prerequisites

- Parts I–IV (especially Chapter 14 assumes familiarity with CUDA events from Part II)
- PyTorch ≥ 2.0, NumPy
- GPU not required (all exercises have CPU fallbacks)

## Exercise Index

| File | What You Learn |
|------|---------------|
| `14.1_benchmark_mistakes.py` | `gather_timings()`, CUDA events, `burst_vs_sustained()` |
| `14.2_throughput_latency_curve.py` | `tl_sweep()`, `find_knee()`, M/D/1 queueing, `benchmark_report()` |
| `14.3_workload_characterization.py` | Arithmetic intensity, roofline position, JSON characterisation report |
| `15.1_porting_checklist.py` | `safe_from_numpy()`, H2D bandwidth, `validate_port()` |
| `15.2_bottleneck_shift.py` | Optimisation ladder, Python overhead floor, scaling ratio |
| `15.3_dataloader_at_scale.py` | `sweep_num_workers()`, `measure_idle_pct()`, CPU affinity |
