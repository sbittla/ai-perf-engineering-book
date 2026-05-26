# Chapter 11 — Batching Strategies

Three exercises showing how the batching strategy determines whether your GPU serves 4 or 40 concurrent users.

## Exercises

| File | Topic | Key Skill |
|------|-------|-----------|
| `11.1_static_batching.py` | Padding waste, GPU idle time | Measure padding fraction; simulate throughput degradation |
| `11.2_continuous_batching.py` | Slot scheduling, throughput comparison | Implement scheduler; quantify throughput × latency improvement |
| `11.3_paged_attention.py` | Block allocator, fragmentation | Implement PagedAllocator; compare static vs paged concurrency |

## The Core Insight

Static batching reserves `max_seq_len` KV memory per request — even if the response is 10 tokens. With variable output lengths, 50–75% of GPU decode work is wasted on padding.

Continuous batching + PagedAttention eliminates both forms of waste:
- **No padding** (continuous batching): GPU always processes real tokens.
- **No fragmentation** (PagedAttention): KV memory allocated on demand, freed immediately.

## Run Order

```bash
python 11.1_static_batching.py      # see the padding problem
python 11.2_continuous_batching.py  # see the fix
python 11.3_paged_attention.py      # see how vLLM manages memory
```

## vLLM Commands

```bash
# Start a vLLM server with continuous batching + PagedAttention
vllm serve meta-llama/Llama-2-7b-chat-hf \
    --max-num-seqs 64 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching
```
