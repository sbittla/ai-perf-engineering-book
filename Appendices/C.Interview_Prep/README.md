# Appendix C — Interview Preparation

50 interview questions with model answers for AI systems performance engineering roles.

## Exercises

| File | Topic | Key Functions |
|------|-------|--------------|
| `C.1_interview_questions.py` | 50 Q&A across 5 categories | `print_questions()`, `--quiz`, `--cat` |

## Workflow

```bash
# Print all 50 questions with answers
python C.1_interview_questions.py

# Self-quiz mode (answers hidden)
python C.1_interview_questions.py --quiz

# One category at a time
python C.1_interview_questions.py --cat 1   # GPU Architecture
python C.1_interview_questions.py --cat 2   # LLM Inference Systems
python C.1_interview_questions.py --cat 3   # Profiling Tools
python C.1_interview_questions.py --cat 4   # Distributed Systems
python C.1_interview_questions.py --cat 5   # Benchmarking
```

## Question Categories

### Category 1 — GPU Architecture (Q1–10)
Warp divergence, compute vs memory-bound, roofline model, SM occupancy,
memory coalescing, Tensor Cores, cache hierarchy, SM utilisation vs occupancy,
PCIe bottlenecks, TF32.

### Category 2 — LLM Inference Systems (Q11–20)
Prefill vs decode phases, KV cache memory formula, continuous batching,
PagedAttention, speculative decoding, precision tradeoffs (FP16/INT8/INT4),
decode throughput limits, decode latency vs context length, chunked prefill,
TTFT vs TPS vs E2E latency.

### Category 3 — Profiling Tools (Q21–30)
nsys vs ncu vs torch.profiler selection, record_function and NVTX annotation,
self_cuda_time_total, profiler schedule pattern, differential flamegraphs,
NVTX in nsys timelines, iowait diagnosis, GPU idle_pct measurement,
torch.utils.bottleneck, the .item() synchronisation trap.

### Category 4 — Distributed Systems (Q31–40)
NCCL importance and benchmarking, tensor vs pipeline parallelism tradeoffs,
FSDP vs DDP, communication/computation overlap, pipeline bubble reduction,
NCCL collectives by parallelism strategy, inference vs training parallelism,
slow NCCL diagnosis, gradient checkpointing, sync vs async AllReduce.

### Category 5 — Benchmarking (Q41–50)
The 5 benchmarking mistakes, throughput-latency curves, arithmetic intensity
measurement, workload characterisation protocol, synthetic vs production
benchmarks, correct speedup reporting, Python overhead floor, thermal
throttling detection, M/D/1 queuing model, torch.compile benchmarking.

## Suggested Study Plan

| Week | Focus | Target |
|------|-------|--------|
| 1 | GPU Architecture (Cat 1) | Explain roofline and occupancy from memory |
| 2 | LLM Inference (Cat 2) | Derive KV cache formula; explain continuous batching |
| 3 | Profiling Tools (Cat 3) | Run nsys and torch.profiler on a real script |
| 4 | Distributed Systems (Cat 4) | Explain FSDP vs DDP memory tradeoff |
| 5 | Benchmarking (Cat 5) | Build a T-L curve from scratch |
| 6 | Full quiz | `--quiz` mode: aim for 40/50 confident answers |

Source reference: `docs/INTERVIEW_PREP.md`
