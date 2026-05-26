# Chapter 18 — Capstone 3: KV Cache Memory Pressure Experiment

A quantitative analysis of LLM KV cache memory behaviour: from the fundamental formula to production deployment planning.

## Exercises

| File | Topic | Key Functions |
|------|-------|---------------|
| `18.1_kv_cache_scaling.py` | KV cache formula, OOM threshold, latency model, strategies | `kv_cache_bytes()`, `max_concurrent_requests()`, `decode_latency_ms()`, `kv_strategy_analysis()` |
| `18.2_memory_budget_planning.py` | Full budget decomposition; 3-way trade-off; deployment plans | `memory_budget()`, `tradeoff_analysis()`, `deployment_plan()` |

## Workflow

```bash
python 18.1_kv_cache_scaling.py    # saves /tmp/capstone18_kv_profile.json
python 18.2_memory_budget_planning.py  # saves /tmp/capstone18_planning_report.json
```

## The KV Cache Formula

```
KV bytes = 2 × n_layers × n_heads × head_dim × seq_len × batch × dtype_bytes
```

**Llama-7B FP16 examples:**
- Per token, per request: 512 KB
- 2048-token context, 1 request: 1 GB
- 2048-token context, 8 requests: 8 GB (plus ~14 GB weights = 22 GB on a 24 GB GPU)

## The 3-Way Trade-off

| Goal | Lever | Cost |
|------|-------|------|
| Higher quality | FP16 instead of INT4 | 4× more weight memory → fewer concurrent requests |
| Longer context | Higher seq_len | More KV cache → fewer concurrent requests |
| More scale | More concurrent requests | More KV cache → shorter context budget |

## KV Cache Strategies

| Strategy | Memory | Quality | Max Concurrent |
|----------|--------|---------|----------------|
| Full cache | O(seq × batch) | Perfect | Limited by VRAM |
| Sliding window | O(window × batch) | Loses distant context | Higher |
| PagedAttention | O(seq × batch) | Perfect (no eviction) | Same as full, but better fragmentation |

## Decode Latency Model

```python
bytes_per_step = 2 × n_layers × n_heads × head_dim × seq_len × dtype_bytes
latency_ms = bytes_per_step / (bandwidth_gb_s × 1e9) × 1000 + base_ms
```

Decode latency scales **linearly** with context length — each step loads all past K,V tensors.
