"""
Exercise 10.1 — From RNNs to Transformers: How LLMs Got Here

Chapter 10 (Background): The Evolution of Large Language Models
Book: AI Systems Performance Engineering

Run:
    python 10.1_llm_evolution.py

What this exercise does:
  1. Explains why sequence modelling is hard and how context determines meaning.
  2. Shows why RNNs fail: vanishing gradients and sequential (5-10% GPU util) compute.
  3. Implements Scaled Dot-Product Attention from scratch on a toy sequence.
  4. Calculates the O(N²) attention memory growth and shows where OOM occurs.
  5. Traces the GPT-1 through LLaMA-3 scaling law timeline.
  6. Calculates maximum decode tokens/sec per GPU based on HBM bandwidth.
  7. Maps the full production inference stack: tokenise → queue → prefill → decode → detokenise.

No TODOs here — this is a read-and-run exercise. Read each section's output,
understand what it means, then explore the chapters in Part IV for hands-on work.
"""

import sys
import math
import time

try:
    import torch
    import torch.nn.functional as F
    TORCH = True
except ImportError:
    TORCH = False
    print("PyTorch not available — using pure Python for numerical sections.")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: The Sequence Problem — Why Context Matters
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 1: The Sequence Problem")
print("=" * 60)

print("""
Natural language is sequential — words depend on what came before.

  "The trophy didn't fit in the suitcase because it was too big."

  'it' refers to the trophy, not the suitcase.
  A model must look ~10 tokens back to resolve this correctly.

  "The bank on the river bank charged interest on my bank account."

  'bank' appears 3 times with 3 different meanings. Context determines
  which meaning is correct at each position.

The challenge for machine learning:
  1. Variable-length sequences (2 words to 100,000 words)
  2. Long-range dependencies (pronoun 500 tokens after its antecedent)
  3. Parallel training (can't wait for word N to process word N+1)
""")

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: The RNN Approach and Its Limits
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 2: RNNs — Good Idea, Wrong Execution")
print("=" * 60)

print("""
Recurrent Neural Networks (RNNs, 1986) process sequences step by step:

  h_0 → [RNN cell] → h_1 → [RNN cell] → h_2 → ... → h_N → output

  h_t = tanh(W_h * h_{t-1} + W_x * x_t + b)

The hidden state h_t is supposed to carry everything the model needs
to remember about the past. In practice, two problems arise:

Problem 1: Vanishing Gradients
  During backpropagation through 512 time steps, the gradient is
  multiplied by the same weight matrix 512 times.
  If max eigenvalue of W < 1: gradient → 0  (vanishes, model forgets)
  If max eigenvalue of W > 1: gradient → ∞  (explodes, training crashes)

  LSTMs (1997) add gates to control what to remember and what to forget.
  They partially fix vanishing gradients but not the sequential constraint.

Problem 2: Sequential Computation
  Step t cannot begin until step t-1 finishes.
  On a GPU with 6,912 CUDA cores, you are using ONE of them per step.
  GPU utilisation during RNN training: typically 5-10%.

  For a 1,000-token sequence with a 1,024-dim hidden state:
    RNN:         1,000 sequential matrix-vector multiplications
    Transformer: 1 parallel matrix-matrix multiplication (all tokens at once)
""")

# Show why sequential kills GPU utilisation
print("Concrete GPU utilisation numbers:")
print(f"  Typical RNN training GPU utilisation  :  5-10%")
print(f"  Typical Transformer training GPU util : 40-90%")
print(f"  Speedup from parallelism alone        : 5-18×")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Attention Mechanism — The Key Insight
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 3: Attention — Learning What to Look At")
print("=" * 60)

print("""
Bahdanau et al. (2014) introduced attention for neural machine translation.
Instead of encoding a whole sentence into one vector, use ALL encoder states
and learn to weight them differently for each decoder step.

  context_t = Σ α_{t,s} * h_s    for all source positions s

  α_{t,s} = softmax(score(h_t, h_s))  — learned alignment weights

Vaswani et al. (2017) "Attention Is All You Need" extended this:
  - NO recurrence at all — pure attention
  - Self-attention: every token attends to every other token
  - Multi-head attention: run H attention heads in parallel

Scaled Dot-Product Attention (the core equation):

  Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V

  Q = queries  (what am I looking for?)
  K = keys     (what do I have to offer?)
  V = values   (what information do I carry?)
  d_k = key dimension (scaling prevents softmax saturation)
""")

if TORCH:
    print("Live demo — minimal attention on a toy sequence:")
    torch.manual_seed(42)
    seq_len, d_model = 8, 16
    d_k = d_model

    Q = torch.randn(seq_len, d_model)
    K = torch.randn(seq_len, d_model)
    V = torch.randn(seq_len, d_model)

    scores = Q @ K.T / math.sqrt(d_k)   # (seq_len, seq_len)
    weights = F.softmax(scores, dim=-1)  # attention weights
    output = weights @ V                 # (seq_len, d_model)

    print(f"  Input shape   : Q/K/V = {tuple(Q.shape)}")
    print(f"  Scores shape  : {tuple(scores.shape)}  (every token × every token)")
    print(f"  Weights shape : {tuple(weights.shape)}  (sum to 1.0 per row)")
    print(f"  Output shape  : {tuple(output.shape)}")
    print(f"  Row 0 weights : {weights[0].detach().numpy().round(3)}")
    print(f"  (token 0 attends to all {seq_len} positions, learned weights)")
else:
    print("  [PyTorch not available — install torch to run the live demo]")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Section 4: O(N²) — The Attention Scaling Problem
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 4: O(N²) Memory — The Scaling Problem")
print("=" * 60)

print("""
The attention matrix (seq_len × seq_len) must be held in GPU memory.
For long sequences this grows quadratically:
""")

d_model = 4096
num_heads = 32
bytes_per_element = 2  # FP16

print(f"  {'Seq Len':>10} {'Attn Matrix (FP16)':>22} {'Fits on H100 80GB?':>20}")
print(f"  {'-'*10} {'-'*22} {'-'*20}")
for seq_len in [512, 2048, 8192, 32768, 131072]:
    mb = seq_len * seq_len * bytes_per_element / 1024 / 1024
    fits = "YES" if mb < 80 * 1024 else "NO (OOM)"
    print(f"  {seq_len:>10,} {mb:>18.1f} MB   {fits:>20}")

print()
print("FlashAttention (Dao et al., 2022) computes attention in tiles,")
print("never materialising the full N×N matrix in HBM.")
print("Memory: O(N²) → O(N). Speed: 2-4× faster on long sequences.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Scaling Laws and the Model Size Explosion
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 5: Scaling Laws — Why Models Got So Large")
print("=" * 60)

models = [
    ("GPT-1",    "2018",  "117M",    "5B",     "First large Transformer LM"),
    ("GPT-2",    "2019",  "1.5B",    "40B",    "Surprising zero-shot ability"),
    ("GPT-3",    "2020",  "175B",    "300B",   "In-context learning emerged"),
    ("PaLM",     "2022",  "540B",    "780B",   "Chain-of-thought reasoning"),
    ("LLaMA-2",  "2023",  "70B",     "2T",     "Open weights, Chinchilla-optimal"),
    ("GPT-4",    "2023",  "~1.8T?",  "unknown","Multimodal, tool use"),
    ("LLaMA-3",  "2024",  "405B",    "15T",    "Long context, 128K tokens"),
]

print(f"  {'Model':<12} {'Year':<6} {'Params':<10} {'Tokens':<10} {'Key Contribution'}")
print(f"  {'-'*12} {'-'*6} {'-'*10} {'-'*10} {'-'*35}")
for name, year, params, tokens, contrib in models:
    print(f"  {name:<12} {year:<6} {params:<10} {tokens:<10} {contrib}")

print()
print("Kaplan et al. (2020) Scaling Laws: loss ∝ (params)^0.076 + (tokens)^0.095")
print("Hoffman et al. (2022) Chinchilla: optimal tokens ≈ 20 × model_parameters")
print("Implication: GPT-3 was under-trained. A 70B model on 1.4T tokens beats it.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Section 6: The Decode Bottleneck — Why Inference Is Hard
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 6: The Decode Bottleneck")
print("=" * 60)

print("""
Training is embarrassingly parallel — process a full batch simultaneously.
Inference decode is inherently sequential — generate ONE token at a time.

  Token 1: read all 140 GB of model weights from HBM → produce 1 token
  Token 2: read all 140 GB of model weights from HBM → produce 1 token
  Token N: read all 140 GB of model weights from HBM → produce 1 token

For a 70B FP16 model:
  Weight size = 70 × 10^9 × 2 bytes = 140 GB
""")

model_gb = 140.0

gpus = [
    ("V100 SXM2",  "2017",  0.9),
    ("A100 SXM4",  "2020",  2.0),
    ("H100 SXM5",  "2022",  3.35),
    ("H200 SXM5",  "2024",  4.8),
]

print(f"  {'GPU':<14} {'Year':<6} {'HBM BW (TB/s)':<16} {'Max tokens/s (1 stream)'}")
print(f"  {'-'*14} {'-'*6} {'-'*16} {'-'*25}")
for gpu, year, bw_tbs in gpus:
    tps = (bw_tbs * 1e12) / (model_gb * 1e9)
    print(f"  {gpu:<14} {year:<6} {bw_tbs:<16.2f} {tps:<25.1f}")

print()
print("Arithmetic intensity of decode: ~2 FLOPs / byte (pure memory-bound).")
print("This is why decode throughput scales with HBM bandwidth, not TFLOP/s.")
print()
print("Three techniques that help:")
print("  KV Cache    — store past key/value projections, avoid recomputing")
print("  Batching    — serve multiple users simultaneously, amortise weight reads")
print("  Speculative — small draft model proposes tokens, large model verifies")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Section 7: The Production LLM Stack
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 7: The Full Production Inference Stack")
print("=" * 60)

stack = [
    ("1. Tokenise",     "CPU",  "Convert text → token IDs (tiktoken/SentencePiece)"),
    ("2. Queue",        "CPU",  "Request batching, priority scheduling, SLA tracking"),
    ("3. Prefill",      "GPU",  "Process all prompt tokens in one forward pass (parallel)"),
    ("4. Decode",       "GPU",  "Autoregressively generate tokens one at a time"),
    ("5. Detokenise",   "CPU",  "Convert token IDs → text, stream to client"),
]

print(f"  {'Stage':<18} {'Where':<8} {'What happens'}")
print(f"  {'-'*18} {'-'*8} {'-'*48}")
for stage, where, what in stack:
    print(f"  {stage:<18} {where:<8} {what}")

print()
print("Prefill is compute-bound (high arithmetic intensity, uses Tensor Cores).")
print("Decode is memory-bandwidth-bound (low arithmetic intensity, HBM limited).")
print("This is why Time-to-First-Token (TTFT) and tokens-per-second (TPS)")
print("require different optimisation strategies.")
print()
print("Explore next: IV.LLM_Inference_Systems/10.LLM_Inference_Fundamentals/")
print("              IV.LLM_Inference_Systems/11.Batching_Strategies/")
