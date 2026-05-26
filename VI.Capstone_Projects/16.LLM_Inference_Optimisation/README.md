# Chapter 16 — Capstone 1: LLM Inference Optimisation Lab

A complete end-to-end optimisation workflow: establish a baseline, apply precision and compile optimisations, then audit the result with torch.profiler.

## Exercises

| File | Topic | Key Functions |
|------|-------|---------------|
| `16.1_baseline_inference.py` | GPT-2 style transformer; measure TTFT, TPS, VRAM | `measure_ttft()`, `measure_tps()`, `save_baseline()` |
| `16.2_precision_and_compile.py` | FP16, BF16, torch.compile optimisation ladder | `measure_precision_speedup()`, `measure_compile_overhead()` |
| `16.3_profiling_audit.py` | torch.profiler op-level attribution; diagnosis-to-fix table | `profile_region_breakdown()` |

## Workflow

```bash
python 16.1_baseline_inference.py    # establishes /tmp/capstone16_baseline.json
python 16.2_precision_and_compile.py  # applies optimisations; saves ladder
python 16.3_profiling_audit.py        # attributes time to ops; saves audit
```

## Key Concepts

**Phases:**
- Prefill (TTFT): compute-bound — one forward pass over all prompt tokens
- Decode (TPS): memory-bandwidth-bound — one forward per generated token

**Optimisation ladder:**
1. FP32 eager (baseline)
2. FP16 eager — halves memory; activates Tensor Cores on GPU
3. FP32 + torch.compile — fuses kernels, reduces dispatch count
4. FP16 + torch.compile — combines both benefits

**Profiler usage:**
```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(5):
        model(idx)

# Sort by CUDA time; read "Self CUDA %" to find real hotspots
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

**compile overhead amortisation:**
```
n_calls_to_break_even = compile_ms / (eager_ms - compiled_ms)
```
For a typical 30s compile and 1ms saving per call: need ~30,000 calls to break even.
