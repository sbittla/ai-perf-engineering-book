# Appendices

Three reference appendices for the AI Systems Performance Engineering book.

## Appendix Map

| Appendix | Topic | File |
|----------|-------|------|
| A — Environment Setup | Verify the full dev environment | `A.1_environment_check.py` |
| B — Command Reference | Profiling cheatsheet + torch.profiler demo | `B.1_profiling_cheatsheet.py` |
| C — Interview Prep | 50 Q&A across 5 categories | `C.1_interview_questions.py` |

## Quick Start

```bash
# Appendix A: verify your environment is ready
python A.Environment_Setup/A.1_environment_check.py

# Appendix B: print cheatsheet and run profiler demo
python B.Command_Reference/B.1_profiling_cheatsheet.py

# Appendix C: print all 50 Q&A
python C.Interview_Prep/C.1_interview_questions.py

# Self-quiz (answers hidden):
python C.Interview_Prep/C.1_interview_questions.py --quiz

# One category at a time (1=GPU Arch, 2=LLM, 3=Profiling, 4=Distributed, 5=Benchmarking):
python C.Interview_Prep/C.1_interview_questions.py --cat 2
```

## Appendix A — Environment Setup

Step-by-step verification of every component required by this book. Runs checks
in five sections: Python/PyTorch versions, GPU capabilities, required packages,
external profiling tools (nsys, ncu, py-spy, perf), and a live timing baseline.

Passes on CPU-only machines; GPU-specific checks are skipped gracefully.

Source reference: `docs/HARDWARE_SETUP.md`

## Appendix B — Command Reference

An interactive cheatsheet that:
1. Prints the diagnostic decision tree (which tool to use first)
2. Demonstrates `torch.profiler` with `record_function` on a small model
3. Compares CUDA event timing vs wall-clock timing
4. Prints the full command reference grouped by tool

Source reference: `docs/COMMANDS.md`

## Appendix C — Interview Preparation

50 questions with detailed model answers, covering:

| Category | Questions | Key Topics |
|----------|-----------|------------|
| 1 — GPU Architecture | Q1–10 | Warps, roofline, occupancy, Tensor Cores, coalescing, TF32 |
| 2 — LLM Inference Systems | Q11–20 | KV cache, continuous batching, speculative decoding, TTFT, TPS |
| 3 — Profiling Tools | Q21–30 | nsys vs ncu vs profiler, NVTX, .item() trap, idle_pct |
| 4 — Distributed Systems | Q31–40 | NCCL, FSDP vs DDP, tensor/pipeline parallel, gradient checkpointing |
| 5 — Benchmarking | Q41–50 | 5 mistakes, throughput-latency curves, arithmetic intensity, thermal throttling |

Source reference: `docs/INTERVIEW_PREP.md`
