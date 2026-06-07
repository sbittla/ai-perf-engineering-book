# Part IV — LLM Inference Systems

> Chapters 10–13 | 13 exercises

This part covers the performance engineering of large language model inference systems: from the fundamental prefill/decode asymmetry through batching strategies, speculative decoding, and distributed inference across multiple GPUs.

## Chapter Map

| Chapter | Topic | Exercises |
|---------|-------|-----------|
| 10 | LLM Inference Fundamentals | 10.1 · 10.2 · 10.3 · 10.4 |
| 11 | Batching Strategies | 11.1 · 11.2 · 11.3 |
| 12 | Speculative Decoding | 12.1 · 12.2 |
| 13 | Distributed Inference | 13.1 · 13.2 · 13.3 · 13.4 |

## Quick Start

```bash
# Chapter 10 — LLM Inference Fundamentals
python "IV.LLM_Inference_Systems/10.LLM_Inference_Fundamentals/10.1_llm_evolution.py"
python "IV.LLM_Inference_Systems/10.LLM_Inference_Fundamentals/10.2_prefill_and_decode.py"
python "IV.LLM_Inference_Systems/10.LLM_Inference_Fundamentals/10.3_kv_cache.py"
python "IV.LLM_Inference_Systems/10.LLM_Inference_Fundamentals/10.4_inference_metrics.py"

# Chapter 11 — Batching Strategies
python "IV.LLM_Inference_Systems/11.Batching_Strategies/11.1_static_batching.py"
python "IV.LLM_Inference_Systems/11.Batching_Strategies/11.2_continuous_batching.py"
python "IV.LLM_Inference_Systems/11.Batching_Strategies/11.3_paged_attention.py"

# Chapter 12 — Speculative Decoding
python "IV.LLM_Inference_Systems/12.Speculative_Decoding/12.1_draft_target_model.py"
python "IV.LLM_Inference_Systems/12.Speculative_Decoding/12.2_acceptance_rate.py"

# Chapter 13 — Distributed Inference
python "IV.LLM_Inference_Systems/13.Distributed_Inference/13.1_tensor_parallelism.py"
python "IV.LLM_Inference_Systems/13.Distributed_Inference/13.2_nccl_collectives.py"
python "IV.LLM_Inference_Systems/13.Distributed_Inference/13.3_fsdp_and_pipeline.py"
python "IV.LLM_Inference_Systems/13.Distributed_Inference/13.4_distributed_simulation.py"
```

## Prerequisites

All exercises run on CPU-only machines. GPU sections are automatically skipped when CUDA is unavailable. The math-heavy sections (KV cache, metrics, speculative decoding, NCCL) need only standard Python and PyTorch.

```bash
pip install torch
```

## Key Concepts

### Prefill vs Decode (Ch 10)
- **Prefill**: processes the entire prompt in one forward pass. Compute-bound (large matmuls). Cost scales as O(seq_len²).
- **Decode**: generates one token per step using the KV cache. Memory-bandwidth-bound. Cost scales as O(seq_len) per token.
- **TTFT** (Time To First Token) = prefill time. **TPS** (Tokens Per Second) = decode throughput.

### KV Cache (Ch 10)
```
KV_bytes = 2 × n_layers × n_heads × head_dim × seq_len × batch × dtype_bytes
```
For Llama-7B FP16: 512 KB per token. A 2048-token context uses 1 GB per request.

### Batching Strategies (Ch 11)
| Strategy | Padding Waste | Concurrency | Latency |
|----------|--------------|-------------|---------|
| Static (HF generate) | High (50–75%) | Low | Predictable |
| Continuous (vLLM) | None | High | Lower P99 |
| + PagedAttention | None | 2–4× more | Lower P99 |

### Speculative Decoding (Ch 12)
- Expected tokens per target step: `E[tok] = (1 - α^(K+1)) / (1 - α)`
- Worthwhile when: α > 0.6 and draft model is 5–20× smaller than target.

### Distributed Inference (Ch 13)
| Strategy | What is sharded | Communication | Use when |
|----------|----------------|---------------|----------|
| Tensor Parallel | Weight matrices | AllReduce per layer | Within node (NVLink) |
| Pipeline Parallel | Layers to stages | Activation passing | Across nodes |
| FSDP | Params + grads + optimizer | AllGather + ReduceScatter | Training / fine-tuning |
