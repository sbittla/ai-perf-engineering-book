# Phase 4 — AI Inference Systems & Distributed Runtime

**Duration: 4 weeks**

## Module 8: LLM Inference Systems
```bash
python module8_llm_systems/infer.py --bench
python module8_llm_systems/kv_cache_sim.py --estimate
python module8_llm_systems/speculative_decode.py --K 4
python module8_llm_systems/continuous_batching.py --exp compare
python module8_llm_systems/serve.py &   # then attach profilers
```

## Module 9: Distributed AI
```bash
python module9_distributed/infer_distributed.py
torchrun --nproc_per_node=2 module9_distributed/nccl_bench.py
python module9_distributed/fsdp_train.py
```
