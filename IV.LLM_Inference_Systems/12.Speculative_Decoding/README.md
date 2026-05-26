# Chapter 12 — Speculative Decoding

Two exercises showing how to use a small draft model to get multiple tokens per target-model step — turning bandwidth-limited decode into a compute-bound verify pass.

## Exercises

| File | Topic | Key Skill |
|------|-------|-----------|
| `12.1_draft_target_model.py` | Draft/verify loop, rejection sampling, speedup formula | Implement `expected_tokens_per_step`; measure vs theory |
| `12.2_acceptance_rate.py` | α measurement, optimal K, diagnosis | Compute α from distributions; find optimal K; build diagnosis table |

## The Speedup Formula

```
E[tokens per target step] = (1 - α^(K+1)) / (1 - α)
```

| α | K=3 | K=5 | K=8 |
|---|-----|-----|-----|
| 0.5 | 1.75 | 1.97 | 2.00 |
| 0.7 | 2.20 | 2.69 | 2.83 |
| 0.9 | 2.87 | 3.57 | 4.09 |

**Only beneficial when**: α > 0.6 AND draft model is 5–20× smaller than target.

## Run Order

```bash
python 12.1_draft_target_model.py   # understand draft/verify mechanics
python 12.2_acceptance_rate.py      # diagnose and tune K
```

## vLLM Command

```bash
vllm serve meta-llama/Llama-2-7b-chat-hf \
    --speculative-model meta-llama/Llama-68M-v1 \
    --num-speculative-tokens 5 \
    --use-v2-block-manager
```
