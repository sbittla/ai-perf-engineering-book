#!/usr/bin/env python3
"""
IV.LLM_Inference_Systems/12.Speculative_Decoding/12.2_acceptance_rate.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chapter 12: Speculative Decoding — Section 2: Acceptance Rate Analysis
=======================================================================
Covers book section 12.2:
  • What drives acceptance rate α: distribution overlap between draft and target
  • Measuring α across different content types (code, creative, factual)
  • K selection: larger K helps when α is high, hurts when α is low
  • Optimal K given a measured α and draft cost
  • Diagnosing poor acceptance rate and remediation strategies

Run:  python IV.LLM_Inference_Systems/12.Speculative_Decoding/12.2_acceptance_rate.py
All sections must print ✓.
"""

import math
import random
import torch
import torch.nn.functional as F

print("=" * 60)
print("  Exercise 12.2 — Acceptance Rate Analysis")
print("=" * 60)
print()


# ─────────────────────────────────────────────────────────────
# SECTION 1: What drives acceptance rate
# ─────────────────────────────────────────────────────────────
print("── Section 1: Acceptance Rate — What It Measures ──")
print("""
  The acceptance rate α measures how often the draft model predicts
  a token that the target model would also have chosen.

  α = E[min(1, p_target[tok] / p_draft[tok])]

  This is the expected ratio of probabilities, not just agreement on the
  argmax token. The rejection sampling rule is conservative — it accepts
  even suboptimal draft tokens if the ratio is high enough.

  WHAT DRIVES α:
    • Vocabulary alignment: draft and target share the same tokeniser (required)
    • Training data similarity: draft trained on same distribution as target
    • Task difficulty: repetitive/structured text → high α; creative/random → low α
    • Temperature: lower temperature → more peaked distributions → higher α
    • Prompt style: few-shot examples align draft with target → higher α

  TYPICAL ACCEPTANCE RATES BY TASK:
    Code completion            : α ≈ 0.75–0.90  (high — structured syntax)
    Document summarisation     : α ≈ 0.65–0.80  (medium — constrained output)
    Creative writing           : α ≈ 0.30–0.60  (low — many valid continuations)
    Open-ended chat            : α ≈ 0.50–0.75  (varies by topic)
    Translation                : α ≈ 0.70–0.85  (high — strong target correspondence)
""")
print("  ✓ Section 1 passed — acceptance rate reflects distribution overlap")


# ─────────────────────────────────────────────────────────────
# SECTION 2: Measuring α from probability distributions
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: Computing Acceptance Rate from Distributions ──")
print("""
  Given draft distribution P_d and target distribution P_t over V tokens,
  the expected acceptance rate for a single token is:

    α = sum over tokens t: P_d[t] × min(1, P_t[t] / P_d[t])
      = sum over tokens t: min(P_d[t], P_t[t])

  This equals 1 minus the total variation distance between P_d and P_t:
    α = 1 - TV(P_d, P_t)

  TODO 1: Implement acceptance_rate_from_distributions() below.
""")


def acceptance_rate_from_distributions(p_draft: torch.Tensor,
                                       p_target: torch.Tensor) -> float:
    """
    TODO 1: Compute expected acceptance rate for one token position.
    = sum_t min(p_draft[t], p_target[t])
    Both inputs are 1D probability vectors summing to 1.
    """
    # YOUR CODE HERE
    return torch.minimum(p_draft, p_target).sum().item()


# Verify: identical distributions → α = 1.0
V = 50
uniform = torch.ones(V) / V
assert abs(acceptance_rate_from_distributions(uniform, uniform) - 1.0) < 1e-5

# Verify: completely disjoint → α = 0.0
p_d = torch.zeros(V); p_d[:V//2] = 2.0/V
p_t = torch.zeros(V); p_t[V//2:] = 2.0/V
assert abs(acceptance_rate_from_distributions(p_d, p_t) - 0.0) < 1e-5

# Simulate different content types by varying distribution similarity
torch.manual_seed(42)

content_types = {
    "Code (structured)":     0.05,   # low noise → similar distributions
    "Translation":           0.15,
    "Summarisation":         0.30,
    "Open-ended chat":       0.50,
    "Creative writing":      0.80,   # high noise → different distributions
}

print(f"  {'Content type':<25}  {'Simulated α':>12}  {'E[tok/step] K=5':>16}")
print(f"  {'─'*25}  {'─'*12}  {'─'*16}")
for ctype, noise in content_types.items():
    alphas = []
    for _ in range(200):
        base = F.softmax(torch.randn(V), dim=0)
        draft  = F.softmax(torch.log(base + 1e-9) + torch.randn(V) * noise, dim=0)
        target = F.softmax(torch.log(base + 1e-9) + torch.randn(V) * noise, dim=0)
        alphas.append(acceptance_rate_from_distributions(draft, target))
    avg_alpha = sum(alphas) / len(alphas)
    e_tok = (1 - avg_alpha**6) / (1 - avg_alpha) if avg_alpha < 0.999 else 6.0
    print(f"  {ctype:<25}  {avg_alpha:>12.3f}  {e_tok:>16.2f}")

print("  ✓ Section 2 passed — acceptance rate computed from distributions")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Optimal K selection
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: Choosing the Optimal K ──")
print("""
  For a given acceptance rate α and draft cost ratio r:
    Net speedup(K) = E[tok/step] / (K × r + 1)
                   = [(1 - α^(K+1)) / (1 - α)] / (K × r + 1)

  Beyond some K*, increasing K gives diminishing returns:
    - More draft steps add cost (K × r grows)
    - But E[tok/step] saturates at 1/(1-α)

  TODO 2: Implement optimal_K() that finds the K maximising net_speedup.
  Search K ∈ [1, K_max] and return the K with highest net_speedup.
""")


def net_speedup(alpha: float, K: int, draft_cost_ratio: float) -> float:
    if abs(alpha - 1.0) < 1e-9:
        e_tok = float(K + 1)
    else:
        e_tok = (1.0 - alpha ** (K + 1)) / (1.0 - alpha)
    return e_tok / (K * draft_cost_ratio + 1.0)


def optimal_K(alpha: float, draft_cost_ratio: float, K_max: int = 20) -> int:
    """
    TODO 2: Return the K in [1, K_max] maximising net_speedup(alpha, K, draft_cost_ratio).
    """
    # YOUR CODE HERE
    best_K  = 1
    best_s  = net_speedup(alpha, 1, draft_cost_ratio)
    for K in range(2, K_max + 1):
        s = net_speedup(alpha, K, draft_cost_ratio)
        if s > best_s:
            best_s = s
            best_K = K
    return best_K


DRAFT_COST = 0.1   # draft = 10% of target cost

print(f"  Draft cost ratio = {DRAFT_COST:.0%} of target")
print(f"\n  {'α':>6}  {'Optimal K':>10}  {'Max speedup':>12}  {'Saturates at':>14}")
print(f"  {'─'*6}  {'─'*10}  {'─'*12}  {'─'*14}")
for alpha in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    K_opt = optimal_K(alpha, DRAFT_COST)
    s_opt = net_speedup(alpha, K_opt, DRAFT_COST)
    saturation = 1.0 / (1.0 - alpha) if alpha < 1.0 else float("inf")
    print(f"  {alpha:>6.2f}  {K_opt:>10}  {s_opt:>12.2f}×  {saturation:>14.1f}")

# Verify: higher α → higher optimal K
K_low_alpha  = optimal_K(0.3, DRAFT_COST)
K_high_alpha = optimal_K(0.9, DRAFT_COST)
assert K_high_alpha >= K_low_alpha, \
    f"Higher α should warrant larger K: {K_low_alpha} vs {K_high_alpha}"
print(f"\n  Higher α → larger optimal K: {K_low_alpha} (α=0.3) vs {K_high_alpha} (α=0.9)")
print("  ✓ Section 3 passed — optimal K increases with acceptance rate")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Diagnosing poor acceptance rate
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Diagnosing and Improving Acceptance Rate ──")
print("""
  SYMPTOMS OF LOW α:
    • Spec decode throughput barely above standard decode
    • vllm metric: spec_decode_draft_acceptance_rate < 0.5
    • Many "bonus token from correction" events in debug logs

  DIAGNOSIS CHECKLIST:
    □ Wrong draft model: draft not trained on same data as target
    □ Temperature mismatch: different temp/top-p between draft and target
    □ Task mismatch: creative task with structured-text draft
    □ Prompt mismatch: system prompt changes distribution significantly
    □ K too large: at low α, most tokens rejected early; reduce K

  REMEDIATION OPTIONS:
    1. Use a task-matched draft model (e.g. CodeLlama-7B draft for code)
    2. Lower temperature (increases α on both models)
    3. Reduce K until net speedup > 1× (see table above)
    4. Use ngram lookup instead of draft model (always α=1 for matched text)
    5. Switch to standard decode for tasks where α < 0.5

  TODO 3: Given observed α = 0.45 and K=5, compute net_speedup.
  If < 1.1×, recommend switching to standard decode.
""")

observed_alpha = 0.45
K_current      = 5
s_current      = net_speedup(observed_alpha, K_current, DRAFT_COST)

# TODO 3: Check if net speedup is below threshold
SPEEDUP_THRESHOLD = 1.1
recommendation = "switch to standard decode" if s_current < SPEEDUP_THRESHOLD else "keep speculative"

print(f"  Observed α = {observed_alpha}, K = {K_current}")
print(f"  Net speedup       : {s_current:.2f}×")
print(f"  Recommendation    : {recommendation}")

# Find the optimal K for this α
K_opt = optimal_K(observed_alpha, DRAFT_COST)
s_opt = net_speedup(observed_alpha, K_opt, DRAFT_COST)
print(f"  Best possible K   : {K_opt}  (speedup = {s_opt:.2f}×)")

assert s_current > 0, "speedup should be positive"
if s_opt > 1.1:
    print(f"  Lowering K to {K_opt} gives {s_opt:.2f}× — worth keeping speculative")
else:
    print(f"  Even optimal K gives only {s_opt:.2f}× — consider standard decode for this task")
print("  ✓ Section 4 passed — α-based diagnosis and K selection working")


# ─────────────────────────────────────────────────────────────
# SECTION 5: Measuring α in a real serving system
# ─────────────────────────────────────────────────────────────
print("\n── Section 5: Monitoring Acceptance Rate in Production ──")
print("""
  vLLM exposes Prometheus metrics you can query with:

    curl http://localhost:8000/metrics | grep spec_decode

  Key metrics:
    vllm:spec_decode_draft_acceptance_rate  — rolling average α per minute
    vllm:spec_decode_efficiency             — actual tokens/target-step
    vllm:spec_decode_num_accepted_tokens    — cumulative accepted tokens
    vllm:spec_decode_num_draft_tokens       — cumulative draft tokens

  Simple acceptance rate estimate from vLLM logs:
    α = num_accepted_tokens / num_draft_tokens

  Adaptive K (future feature in vLLM):
    If α drops below 0.6, automatically reduce K.
    If α rises above 0.8, automatically increase K.
    This keeps the system self-tuning to the current content distribution.

  Benchmark command to measure spec decode improvement:
    python benchmarks/benchmark_serving.py \\
        --backend vllm --request-rate 5 --num-prompts 200 \\
        --model <target> \\
        --speculative-model <draft>
    Compare output_token_throughput with and without --speculative-model.
""")
print("  ✓ Section 5 passed — production monitoring strategy understood")


print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 12.2 complete!")
print("  You can measure acceptance rate, find optimal K, diagnose")
print("  poor performance, and monitor spec decode in production.")
print("  Next: IV.LLM_Inference_Systems/13.Distributed_Inference/13.1_tensor_parallelism.py")
print("=" * 60)
