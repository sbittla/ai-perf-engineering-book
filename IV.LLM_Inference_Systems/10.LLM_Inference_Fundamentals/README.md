# Chapter 10 — LLM Inference Fundamentals

Four exercises covering the core concepts every inference engineer must know before optimizing anything.

## Exercises

| File | Topic | Key Skill |
|------|-------|-----------|
| `10.1_llm_evolution.py` | From RNNs to Transformers, the evolution of LLMs | Read-and-run background on how modern LLMs got here |
| `10.2_prefill_and_decode.py` | Prefill vs decode phases, TTFT, TPS | Measure both phases; compute arithmetic intensity |
| `10.3_kv_cache.py` | KV cache formula, memory estimation | Calculate KV memory for any model; find max concurrency |
| `10.4_inference_metrics.py` | P50/P99, throughput vs latency tradeoff | Implement percentile, SLO compliance check, batch sweep |

## The Essential Formula

```
KV_bytes = 2 × n_layers × n_heads × head_dim × seq_len × batch × dtype_bytes
```

For Llama-2-7B FP16: **512 KB per token per request**. Know this cold.

## Run Order

```bash
python 10.1_llm_evolution.py        # background: how LLMs got here
python 10.2_prefill_and_decode.py   # understand the two phases
python 10.3_kv_cache.py             # budget your VRAM
python 10.4_inference_metrics.py    # set and measure SLOs
```
