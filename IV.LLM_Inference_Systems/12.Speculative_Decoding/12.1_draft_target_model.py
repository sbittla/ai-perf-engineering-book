#!/usr/bin/env python3
"""
IV.LLM_Inference_Systems/12.Speculative_Decoding/12.1_draft_target_model.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chapter 12: Speculative Decoding — Section 1: Draft and Target Model
=======================================================================
Covers book section 12.1:
  • The memory-bandwidth bottleneck that speculative decoding addresses
  • Draft model: small, fast, generates K candidate tokens cheaply
  • Target model: large, authoritative, verifies all K tokens in one pass
  • Rejection sampling: ensures output distribution matches target exactly
  • Speedup formula: E[tokens/step] = (1 - α^(K+1)) / (1 - α)
  • When speculative decoding helps vs hurts

Run:  python IV.LLM_Inference_Systems/12.Speculative_Decoding/12.1_draft_target_model.py
All sections must print ✓.
"""

import math
import random
import time
import torch
import torch.nn.functional as F

print("=" * 60)
print("  Exercise 12.1 — Speculative Decoding: Draft + Target")
print("=" * 60)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}\n")


# ─────────────────────────────────────────────────────────────
# SECTION 1: Why decode is slow — and the speculative insight
# ─────────────────────────────────────────────────────────────
print("── Section 1: The Memory-Bandwidth Bottleneck ──")
print("""
  Standard autoregressive decode:
    Each step reads ALL model weights from DRAM to produce ONE token.
    Arithmetic intensity ≈ 1 FLOPs/byte — always memory-bandwidth bound.
    Doubling GPU memory bandwidth → halves decode latency.
    Adding more FLOPs (Tensor Cores) → no improvement for small batches.

  The speculative decoding insight:
    One target-model forward pass with seq_len+K tokens costs almost
    the same memory bandwidth as a pass with seq_len tokens.
    (The extra K tokens just add compute — but we are NOT compute-bound.)

    Therefore:
      Run DRAFT model (5–20× smaller) K times to get K candidate tokens.
      Run TARGET model ONCE to verify all K candidates in parallel.
      If all K accepted: 1 verify pass → K tokens (vs 1 token in standard decode)

  Best case (all accepted): K× throughput improvement.
  With acceptance rate α:   E[tokens/step] = (1 - α^(K+1)) / (1 - α)
""")
print("  ✓ Section 1 passed — speculative decoding exploits bandwidth vs compute asymmetry")


# ─────────────────────────────────────────────────────────────
# SECTION 2: The speedup formula
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: Expected Speedup Formula ──")
print("""
  If the draft model accepts each token with probability α (acceptance rate):

  Expected tokens per target-model step:
      E[tok] = (1 - α^(K+1)) / (1 - α)

  This comes from the geometric series: each accepted draft token has
  probability α^k, and we stop at the first rejection.

  TODO 1: Implement expected_tokens_per_step(alpha, K) below.
""")


def expected_tokens_per_step(alpha: float, K: int) -> float:
    """
    TODO 1: Return expected number of tokens generated per target-model step.
    Formula: (1 - alpha^(K+1)) / (1 - alpha)
    Handle alpha == 1.0 as a special case → return K + 1.
    """
    # YOUR CODE HERE
    if abs(alpha - 1.0) < 1e-9:
        return float(K + 1)
    return (1.0 - alpha ** (K + 1)) / (1.0 - alpha)


# Verify known values
assert abs(expected_tokens_per_step(0.0, 5) - 1.0) < 0.01, "α=0 → always 1 token"
assert abs(expected_tokens_per_step(1.0, 5) - 6.0) < 0.01, "α=1 → K+1=6 tokens"

print(f"  {'K':>4}  {'α=0.3':>8}  {'α=0.5':>8}  {'α=0.7':>8}  {'α=0.9':>8}  {'α=1.0':>8}")
print(f"  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
for K in [1, 3, 5, 8, 12]:
    row = [f"{expected_tokens_per_step(a, K):.2f}" for a in [0.3, 0.5, 0.7, 0.9, 1.0]]
    print(f"  {K:>4}  {row[0]:>8}  {row[1]:>8}  {row[2]:>8}  {row[3]:>8}  {row[4]:>8}")

# At α=0.8, K=5: expected ≈ 3.6
e = expected_tokens_per_step(0.8, 5)
assert abs(e - 3.6) < 0.2, f"Expected ~3.6 at α=0.8 K=5, got {e:.2f}"
print(f"\n  At α=0.8, K=5: E[tok] = {e:.2f}  (3.6× speedup over standard decode)")
print("  ✓ Section 2 passed — speedup formula working")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Simulating draft and target models
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: Draft + Target Model Simulation ──")
print("""
  We simulate draft and target models as probability distributions.
  The draft model generates K tokens; the target model verifies them.
  Rejection sampling ensures the final distribution matches the target.

  REJECTION SAMPLING RULE (per token k):
    Let p_d = P_draft[token_k]   (draft probability for this token)
        p_t = P_target[token_k]  (target probability for this token)
    Accept token_k with probability min(1, p_t / p_d).
    On rejection: sample a correction from max(0, p_target - p_draft).

  This is unbiased: the output distribution is EXACTLY the target distribution,
  regardless of the acceptance rate.
""")

VOCAB = 100   # small vocab for simulation


def make_draft_logits(vocab_size=VOCAB, concentration=1.0):
    """Generate random draft model logits (higher concentration = more peaked)."""
    return torch.randn(vocab_size) * concentration


def make_target_logits(draft_logits, noise=0.3):
    """Target model: correlated with draft but slightly different."""
    return draft_logits + torch.randn_like(draft_logits) * noise


def speculative_step(draft_probs: torch.Tensor, target_probs: torch.Tensor,
                     draft_tokens: torch.Tensor, K: int) -> tuple:
    """
    Run one round of rejection sampling for K draft tokens.
    Returns (n_accepted, accepted_tokens, bonus_token).
    """
    accepted_tokens = []
    for k in range(K):
        tok   = draft_tokens[k].item()
        p_d   = draft_probs[k, tok].item() + 1e-9
        p_t   = target_probs[k, tok].item()
        ratio = min(1.0, p_t / p_d)

        if random.random() < ratio:
            accepted_tokens.append(tok)
        else:
            # Sample correction from max(0, p_target - p_draft)
            correction = torch.clamp(target_probs[k] - draft_probs[k], min=0)
            if correction.sum() > 0:
                correction = correction / correction.sum()
                corrected_tok = torch.multinomial(correction, 1).item()
            else:
                corrected_tok = torch.multinomial(target_probs[k], 1).item()
            accepted_tokens.append(corrected_tok)
            # Stop at first rejection
            n_accepted = len(accepted_tokens) - 1
            bonus_tok  = corrected_tok
            return n_accepted, accepted_tokens[:n_accepted], bonus_tok

    # All K accepted — bonus token from target
    bonus_probs = target_probs[K]
    bonus_tok   = torch.multinomial(bonus_probs, 1).item()
    return K, accepted_tokens, bonus_tok


# Simulate many steps to measure acceptance rate and tokens/step
K_DRAFT = 5
N_STEPS = 1000
random.seed(42)
torch.manual_seed(42)

total_accepted = 0
total_steps    = 0
tokens_per_step_list = []

for _ in range(N_STEPS):
    # Generate K+1 sets of logits (K draft positions + 1 bonus position)
    draft_logits_seq  = [make_draft_logits()          for _ in range(K_DRAFT + 1)]
    target_logits_seq = [make_target_logits(dl)        for dl in draft_logits_seq]

    draft_probs_seq   = torch.stack([F.softmax(l, dim=-1) for l in draft_logits_seq])
    target_probs_seq  = torch.stack([F.softmax(l, dim=-1) for l in target_logits_seq])

    # Sample K draft tokens
    draft_tokens = torch.stack([
        torch.multinomial(draft_probs_seq[k], 1).squeeze() for k in range(K_DRAFT)
    ])

    n_acc, _, _ = speculative_step(draft_probs_seq, target_probs_seq, draft_tokens, K_DRAFT)
    tokens_generated = n_acc + 1   # n_acc accepted + 1 bonus token
    total_accepted  += n_acc
    # Corrected denominator: count tokens actually attempted (stop at rejection)
    total_steps     += n_acc + 1 if n_acc < K_DRAFT else K_DRAFT
    tokens_per_step_list.append(tokens_generated)

# Corrected α: accepted / attempted (not accepted / total_draft_budget)
alpha_measured   = total_accepted / total_steps if total_steps > 0 else 0.0
avg_tok_per_step = sum(tokens_per_step_list) / len(tokens_per_step_list)
theoretical_tok  = expected_tokens_per_step(alpha_measured, K_DRAFT)

print(f"  Simulated {N_STEPS} speculative decode steps (K={K_DRAFT})")
print(f"  Acceptance rate α : {alpha_measured:.3f}")
print(f"  Avg tok/step      : {avg_tok_per_step:.2f}  (measured)")
print(f"  Theoretical       : {theoretical_tok:.2f}  (formula)")

assert abs(avg_tok_per_step - theoretical_tok) < 1.5, \
    f"Measured ({avg_tok_per_step:.2f}) should roughly match theoretical ({theoretical_tok:.2f})"
print("  ✓ Section 3 passed — rejection sampling matches theoretical formula")


# ─────────────────────────────────────────────────────────────
# SECTION 4: When speculative decoding helps vs hurts
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: When It Helps and When It Doesn't ──")
print("""
  HELPS when:
    • Target model is large and memory-bandwidth bound during decode
    • Draft model is 5–20× smaller (fast to run K times)
    • Content is "easy" for draft (code, structured text, repetitive output)
      → high α (0.7–0.9) → large speedup multiplier
    • Batch size is small (decode is BW-bound, not compute-bound)

  HURTS or gives no gain when:
    • Draft model is too large (draft K steps costs more than target saves)
    • Batch size is large (target verify is already compute-bound)
    • Content is creative/random (α ≈ 0.1–0.3) → barely better than 1 tok/step
    • Draft and target have very different distributions (many rejections)

  PRACTICAL RULE OF THUMB:
    Worthwhile if: (draft_size / target_size) < 0.1 AND α > 0.6

  TODO 2: Fill in the table of net speedup at various (α, K) combinations.
  Net speedup accounts for draft overhead: assume draft_cost = 0.1 × target_cost.
""")

DRAFT_COST_RATIO = 0.1   # draft model costs 10% of target model per step


def net_speedup(alpha: float, K: int, draft_cost_ratio: float) -> float:
    """
    TODO 2: Return the net speedup multiplier.
    In one "speculative step" we:
      - Run draft K times: cost = K × draft_cost_ratio  (in units of target cost)
      - Run target once:   cost = 1.0
      Total cost = K * draft_cost_ratio + 1.0
      Tokens produced = E[tok] = expected_tokens_per_step(alpha, K)
      Tokens per target-cost unit = E[tok] / total_cost
      Standard decode tokens per target-cost unit = 1.0
      Net speedup = E[tok] / (K * draft_cost_ratio + 1.0)
    """
    # YOUR CODE HERE
    e_tok = expected_tokens_per_step(alpha, K)
    total_cost = K * draft_cost_ratio + 1.0
    return e_tok / total_cost


print(f"  Draft cost = {DRAFT_COST_RATIO:.0%} of target.  Net speedup (>1.0 = win):")
print(f"\n  {'K':>4}  {'α=0.3':>8}  {'α=0.5':>8}  {'α=0.7':>8}  {'α=0.9':>8}")
print(f"  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
for K in [1, 3, 5, 8]:
    row = [f"{net_speedup(a, K, DRAFT_COST_RATIO):.2f}×" for a in [0.3, 0.5, 0.7, 0.9]]
    print(f"  {K:>4}  {row[0]:>8}  {row[1]:>8}  {row[2]:>8}  {row[3]:>8}")

# Verify α=0.9, K=5 gives speedup > 2
s = net_speedup(0.9, 5, DRAFT_COST_RATIO)
assert s > 2.0, f"Expected >2× at α=0.9 K=5, got {s:.2f}"
print(f"\n  At α=0.9, K=5: net speedup = {s:.2f}×")
print("  ✓ Section 4 passed — net speedup accounts for draft model cost")


# ─────────────────────────────────────────────────────────────
# SECTION 5: vLLM speculative decoding setup
# ─────────────────────────────────────────────────────────────
print("\n── Section 5: Enabling Speculative Decoding in vLLM ──")
print("""
  vLLM supports speculative decoding via:

    vllm serve meta-llama/Llama-2-7b-chat-hf \\
        --speculative-model meta-llama/Llama-68M-v1 \\
        --num-speculative-tokens 5 \\
        --use-v2-block-manager

  Key parameters:
    --speculative-model   : the draft model (must share vocabulary with target)
    --num-speculative-tokens : K draft tokens per step (typically 3–8)
    --use-v2-block-manager   : required for speculative decoding in vLLM

  Measuring acceptance rate in production:
    vllm exposes Prometheus metrics including:
      vllm:spec_decode_draft_acceptance_rate   (rolling average α)
      vllm:spec_decode_efficiency              (effective tok/step)
    If α drops below 0.5, consider reducing K or switching draft models.

  ngram speculative decoding (no separate draft model):
    --speculative-model [ngram] \\
    --num-speculative-tokens 5 \\
    --ngram-prompt-lookup-max 4
    Works well for tasks with high self-repetition (code, structured output).
""")
print("  ✓ Section 5 passed — vLLM speculative decoding parameters understood")


print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 12.1 complete!")
print("  You understand the draft-verify loop, rejection sampling,")
print("  the speedup formula, and when speculative decoding helps.")
print("  Next: IV.LLM_Inference_Systems/12.Speculative_Decoding/12.2_acceptance_rate.py")
print("=" * 60)
