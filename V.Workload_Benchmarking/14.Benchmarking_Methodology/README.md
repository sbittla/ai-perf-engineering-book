# Chapter 14 — Benchmarking Methodology

Three exercises covering the full benchmarking toolkit: avoiding common mistakes, plotting throughput-latency curves, and producing structured workload characterisation reports.

## Exercises

| File | Topic | Key Functions |
|------|-------|---------------|
| `14.1_benchmark_mistakes.py` | The five benchmarking mistakes and their fixes | `gather_timings()`, `burst_vs_sustained()` |
| `14.2_throughput_latency_curve.py` | Throughput-latency curves, queuing theory, Pareto frontier | `tl_sweep()`, `find_knee()`, `md1_latency_ms()`, `benchmark_report()` |
| `14.3_workload_characterization.py` | 8-point characterisation checklist, roofline position, JSON reports | `arithmetic_intensity()`, `measure_peak_bandwidth_gbs()`, `add_summary()` |

## Quick Start

```bash
python 14.1_benchmark_mistakes.py
python 14.2_throughput_latency_curve.py
python 14.3_workload_characterization.py
```

## Key Concepts

### The Five Benchmarking Mistakes (14.1)

| # | Mistake | Fix |
|---|---------|-----|
| 1 | No warmup | Discard first ≥ 5 runs |
| 2 | `time.time()` on GPU | Use CUDA events: `s.elapsed_time(e)` |
| 3 | Single sample | Collect ≥ 50 measurements; report mean ± std + P99 |
| 4 | Multiple variables at once | Change one variable per experiment |
| 5 | Reporting peak burst | Report sustained throughput (10–30 s window) |

### Throughput-Latency Curves (14.2)

- **Knee**: the batch size where P99/mean > threshold (2.0×); beyond it, latency diverges
- **Pareto frontier**: for each latency budget, the highest-throughput configuration
- **M/D/1 queueing**: `W = S + ρ·S / (2·(1-ρ))` where `ρ = λ·S` (utilisation)
- At ρ → 1: latency → ∞; keep ρ < 0.8 for production systems

### Workload Characterisation (14.3)

**Arithmetic intensity of a Linear layer:**
```
FLOPs = 2 × batch × seq × d_out × d_in
Bytes = (d_out×d_in + batch×seq×d_in) × dtype_bytes
AI    = FLOPs / Bytes
```

- A100 ridge point (FP32): ~156 FLOPs/byte
- batch=1 is always memory-bandwidth-bound
- batch=128 with large d_model can be compute-bound

**The 8-Point Checklist:**
1. Peak memory bandwidth utilisation (DRAM%)
2. Peak compute utilisation (SM%)
3. Roofline position
4. Throughput at max sustained load
5. Latency P50/P99 per config
6. GPU memory footprint
7. CPU utilisation during GPU work
8. Thermal throttling check

## Comparing Reports

```python
import json
before = json.load(open("characterisation_report_before.json"))
after  = json.load(open("characterisation_report_after.json"))
for c_b, c_a in zip(before["configs"], after["configs"]):
    speedup = c_a["throughput_sps"] / c_b["throughput_sps"]
    print(f"B={c_b['batch']}: {speedup:.2f}×")
```
