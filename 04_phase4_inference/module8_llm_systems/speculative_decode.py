#!/usr/bin/env python3
"""
speculative_decode.py  ─  Phase 4 / Module 8: Speculative Decoding
====================================================================

HOW TO RUN
    python speculative_decode.py
    python speculative_decode.py --K 5 --steps 50   # K draft tokens


"""

import argparse, os, sys, time
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--K",        type=int, default=4,   help="Draft tokens per step")
parser.add_argument("--steps",    type=int, default=30,  help="Generation steps")
parser.add_argument("--temp",     type=float, default=0.8)
parser.add_argument("--prompt-len", type=int, default=20)
args = parser.parse_args()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'models'))
try:
    from model import TinyTransformer
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def cuda_ms(fn, warmup=2, iters=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

print(f"\n{'='*60}")
print(f"  speculative_decode.py  ─  Phase 4 Module 8")
print(f"  Device: {DEVICE}  K={args.K}  Steps={args.steps}")
print(f"{'='*60}\n")

if not HAS_MODEL:
    print("  model.py not found in expected path. Skipping.")
    exit(0)

# ── Load models ────────────────────────────────────────────────────────────────
# Draft model: SMALL (fast, cheap per-token)
# Target model: LARGE (slow, authoritative)
# Here we use tiny/small as stand-ins for 68M/7B
draft_model  = TinyTransformer("tiny",  max_seq=256).to(DEVICE).eval()
target_model = TinyTransformer("small", max_seq=256).to(DEVICE).eval()

VOCAB = 50257
prompt = torch.randint(0, VOCAB, (1, args.prompt_len), device=DEVICE)

print(f"  Draft  model: TinyTransformer[tiny]   "
      f"({sum(p.numel() for p in draft_model.parameters())/1e6:.1f}M params)")
print(f"  Target model: TinyTransformer[small]  "
      f"({sum(p.numel() for p in target_model.parameters())/1e6:.1f}M params)")
print()

# ─────────────────────────────────────────────────────────────────────────────
# STANDARD DECODE (baseline)
# ─────────────────────────────────────────────────────────────────────────────
def standard_decode(prompt, n_steps):
    """Generate n_steps tokens one at a time using target model only."""
    ids = prompt.clone()
    kv  = None
    for _ in range(n_steps):
        with torch.no_grad():
            logits, kv = target_model(ids[:, -1:] if kv else ids, kv)
        logit = logits[:, -1, :] / args.temp
        next_tok = torch.multinomial(F.softmax(logit, dim=-1), 1)
        ids = torch.cat([ids, next_tok], dim=1)
    return ids

# Warmup
standard_decode(prompt, 3)
t_std = cuda_ms(lambda: standard_decode(prompt, args.steps), warmup=2, iters=3)
std_tps = args.steps / (t_std / 1000)

print(f"  Standard decode ({args.steps} steps): {t_std:.1f}ms  →  {std_tps:.1f} tok/s")

# ─────────────────────────────────────────────────────────────────────────────
# SPECULATIVE DECODE
# ─────────────────────────────────────────────────────────────────────────────
def speculative_decode(prompt, n_steps, K):
    """
    Speculative decoding with K draft tokens per step.

    Returns (output_ids, n_accepted_total, n_steps_used)
    """
    ids             = prompt.clone()
    n_accepted_total = 0
    step            = 0

    while step < n_steps:
        # ── Phase 1: Draft generates K candidate tokens ──────────────────────
        draft_ids = ids.clone()
        draft_kv  = None
        draft_probs = []   # store draft P(token | context) for each of K tokens

        for k in range(K):
            with torch.no_grad():
                logits, draft_kv = draft_model(
                    draft_ids[:, -1:] if draft_kv else draft_ids, draft_kv
                )
            p_draft  = F.softmax(logits[:, -1, :] / args.temp, dim=-1)
            next_tok = torch.multinomial(p_draft, 1)
            draft_probs.append(p_draft)
            draft_ids = torch.cat([draft_ids, next_tok], dim=1)

        # ── Phase 2: Target verifies all K tokens in ONE forward pass ────────
        # Target sees: [original prompt + K draft tokens]
        with torch.no_grad():
            # One forward pass over original context + all K draft tokens
            all_logits, _ = target_model(draft_ids)

        # Target produces K+1 logit vectors (one per new position)
        target_probs = [
            F.softmax(all_logits[:, -(K+1-k), :] / args.temp, dim=-1)
            for k in range(K + 1)
        ]

        # ── Phase 3: Rejection sampling ──────────────────────────────────────
        # Accept/reject each draft token based on ratio of target/draft probs.
        # This ensures final distribution matches target exactly.
        accepted = 0
        for k in range(K):
            draft_tok = draft_ids[:, len(prompt[0]) + k]   # k-th draft token
            p_t = target_probs[k][0, draft_tok].item()
            p_d = draft_probs[k][0, draft_tok].item() + 1e-9
            ratio = min(1.0, p_t / p_d)

            if torch.rand(1).item() < ratio:
                # Accept: keep draft token
                accepted += 1
            else:
                # Reject: sample correction from (target - draft) distribution
                corrected = torch.clamp(target_probs[k] - draft_probs[k], min=0)
                corrected = corrected / corrected.sum()
                correction_tok = torch.multinomial(corrected, 1)
                draft_ids = torch.cat([
                    ids, correction_tok
                ], dim=1)
                break

        # Append accepted tokens + one bonus token from target
        accepted_end = len(prompt[0]) + accepted
        ids = draft_ids[:, :accepted_end]
        # Add one more token from target distribution
        bonus_tok = torch.multinomial(target_probs[accepted], 1)
        ids = torch.cat([ids, bonus_tok], dim=1)

        n_accepted_total += accepted
        step += 1 + accepted   # each spec step produces 1+accepted tokens

    return ids, n_accepted_total, step

# Warmup
speculative_decode(prompt, 5, args.K)

# Timed run
t0 = time.perf_counter()
out_ids, n_accepted, n_spec_steps = speculative_decode(prompt, args.steps, args.K)
torch.cuda.synchronize()
t_spec = (time.perf_counter() - t0) * 1000

n_new_tokens   = out_ids.shape[1] - prompt.shape[1]
spec_tps       = n_new_tokens / (t_spec / 1000)
acceptance_rate = n_accepted / (n_spec_steps * args.K + 1e-9)
speedup         = spec_tps / std_tps

print(f"  Speculative  ({args.K} draft):  {t_spec:.1f}ms  →  {spec_tps:.1f} tok/s")
print(f"\n{'='*60}")
print(f"  RESULTS")
print(f"{'='*60}")
print(f"  K (draft tokens)   : {args.K}")
print(f"  Acceptance rate α  : {acceptance_rate:.2f}")
print(f"  Tokens generated   : {n_new_tokens}")
print(f"  Speculative steps  : {n_spec_steps}")
print(f"  Standard tok/s     : {std_tps:.1f}")
print(f"  Speculative tok/s  : {spec_tps:.1f}")
print(f"  Speedup            : {speedup:.2f}×")

# Theoretical prediction
import math
alpha = acceptance_rate
if alpha < 1:
    theoretical = (1 - alpha**(args.K+1)) / (1 - alpha)
else:
    theoretical = args.K + 1
print(f"  Theoretical E[tok] : {theoretical:.2f} per verify step  (formula: (1-α^(K+1))/(1-α))")

print(f"""
  NOTES
    • Acceptance rate {acceptance_rate:.2f} means draft was right {acceptance_rate*100:.0f}% of the time
    • With random draft/target models, α ≈ 1/vocab_size (nearly 0)
    • Real systems: draft trained on same data → α ≈ 0.7–0.9
    • vLLM supports speculative decoding: --speculative-model <draft_model>
""")
