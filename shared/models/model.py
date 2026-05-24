#!/usr/bin/env python3
"""
model.py  —  Shared Model Definitions (Used by train.py, infer.py, serve.py)
==============================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# =============================================================================
# 1 — TINY TRANSFORMER (GPT-style Causal Language Model)
# =============================================================================

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    CAUSAL = each token can only attend to itself and PREVIOUS tokens.
    This is enforced by the attention mask: positions (i, j) where j > i
    are set to -inf before softmax, so they contribute zero after softmax.

    WHY THIS MATTERS FOR PROFILING:
        - The QKV projection (linear layers) are compute-bound when seq is long.
        - The attention score computation (Q @ K.T) scales O(seq²) — the main
          bottleneck for long-context models (why FlashAttention was invented).
        - The KV cache stores past K and V tensors so we don't recompute them
          on each new token (decode phase: memory-bandwidth bound).

    Parameters:
        d_model   — embedding dimension (e.g. 256, 512, 768)
        n_heads   — number of attention heads (d_model must be divisible by n_heads)
        max_seq   — maximum sequence length (for pre-computing causal mask)
        dropout   — attention dropout probability
    """
    def __init__(self, d_model: int, n_heads: int, max_seq: int = 512, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model  = d_model
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads   # dimension per head

        # Combined QKV projection: one matrix multiply instead of 3
        # Output shape: (B, T, 3 * d_model)  → split into Q, K, V
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)

        # Output projection: recombine all heads back to d_model
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Causal mask: lower-triangular matrix of ones
        # register_buffer: saved with model but not a learnable parameter
        # Shape: (1, 1, max_seq, max_seq) — broadcast over batch and heads
        mask = torch.tril(torch.ones(max_seq, max_seq))
        self.register_buffer("causal_mask", mask.view(1, 1, max_seq, max_seq))

    def forward(
        self,
        x: torch.Tensor,                                    # (B, T, d_model)
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # past (K, V)
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with optional KV cache for autoregressive generation.

        WITHOUT kv_cache (prefill phase or training):
            Process all T tokens together. O(T²) attention.

        WITH kv_cache (decode phase):
            x has only the NEW token (T=1).
            Concatenate new K, V with cached past K, V.
            Attend over the full history without recomputing past K, V.
            This is why decode is memory-bandwidth bound — we load the
            entire KV cache from HBM on every step.
        """
        B, T, C = x.shape  # Batch, Sequence length, Channels (d_model)

        # Compute Q, K, V in one shot, then split
        # qkv shape: (B, T, 3*d_model)
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=2)   # each: (B, T, d_model)

        # Reshape for multi-head attention:
        # (B, T, d_model) → (B, T, n_heads, head_dim) → (B, n_heads, T, head_dim)
        # Transposing puts the seq dim after heads so matmul works correctly
        def reshape_heads(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape_heads(q), reshape_heads(k), reshape_heads(v)

        # ── KV Cache handling ────────────────────────────────────────────────
        # During decode: concatenate new K, V with cached past K, V
        # kv_cache = (past_k, past_v), each shape (B, n_heads, T_past, head_dim)
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)   # (B, n_heads, T_past+1, head_dim)
            v = torch.cat([past_v, v], dim=2)

        new_kv_cache = (k, v)   # Return updated cache for next decode step

        # ── Scaled dot-product attention ─────────────────────────────────────
        # Scale: divide by sqrt(head_dim) to prevent softmax saturation
        # Without scaling, dot products grow with head_dim → vanishing gradients
        T_k = k.shape[2]  # Total key length (includes past cache)
        scale = 1.0 / math.sqrt(self.head_dim)

        # Attention scores: (B, n_heads, T_q, T_k)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask: mask out future positions (upper triangle → -inf)
        # After softmax, -inf → 0 (no attention to future tokens)
        if kv_cache is None:
            # During training/prefill: apply full causal mask
            mask = self.causal_mask[:, :, :T, :T_k]
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum of values: (B, n_heads, T_q, head_dim)
        y = torch.matmul(attn, v)

        # Reassemble heads: (B, n_heads, T, head_dim) → (B, T, d_model)
        y = y.transpose(1, 2).contiguous().view(B, T, self.d_model)

        return self.resid_drop(self.out_proj(y)), new_kv_cache


class TransformerBlock(nn.Module):
    """
    One transformer block: LayerNorm → Attention → residual + LayerNorm → FFN → residual.

    PRE-NORM (norm before attention) is used here because it:
    - Trains more stably than post-norm
    - Is used by GPT-2, Llama, and most modern LLMs
    """
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4,
                 max_seq: int = 512, dropout: float = 0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq, dropout)
        self.ln2  = nn.LayerNorm(d_model)

        # FFN: expand d_model by ffn_mult, apply GELU, project back
        # GELU is smoother than ReLU → better gradient flow → used in GPT-2, BERT
        ffn_dim = d_model * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim, bias=False),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x, kv_cache=None):
        # Pre-norm + residual connection for attention
        attn_out, new_cache = self.attn(self.ln1(x), kv_cache)
        x = x + attn_out                   # residual: preserve gradient flow

        # Pre-norm + residual connection for FFN
        x = x + self.ffn(self.ln2(x))
        return x, new_cache


class TinyTransformer(nn.Module):
    """
    Minimal GPT-style causal language model for profiling experiments.

    Size configurations (d_model, n_heads, n_layers):
        tiny   : (128,  4,  2)  ~1.5M params  — fits on CPU, fast for testing
        small  : (256,  4,  4)  ~7M params    — good for GPU experiments
        medium : (512,  8,  6)  ~45M params   — realistic for profiling
        large  : (768, 12, 12)  ~117M params  — GPT-2 size

    The model:
        token embedding → positional embedding → N transformer blocks → LM head
    """
    CONFIGS = {
        "tiny":   dict(d_model=128,  n_heads=4,  n_layers=2,  vocab_size=50257),
        "small":  dict(d_model=256,  n_heads=4,  n_layers=4,  vocab_size=50257),
        "medium": dict(d_model=512,  n_heads=8,  n_layers=6,  vocab_size=50257),
        "large":  dict(d_model=768,  n_heads=12, n_layers=12, vocab_size=50257),
    }

    def __init__(self, size: str = "small", max_seq: int = 512, dropout: float = 0.1):
        super().__init__()
        cfg = self.CONFIGS[size]
        self.d_model   = cfg["d_model"]
        self.n_layers  = cfg["n_layers"]
        self.vocab_size = cfg["vocab_size"]
        self.max_seq   = max_seq

        # Token embedding: convert token IDs → dense vectors
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["d_model"])

        # Positional embedding: learned position encoding (GPT-2 style)
        # Alternative: RoPE (Llama), ALiBi (MPT) — both avoid this Embedding table
        self.pos_emb = nn.Embedding(max_seq, cfg["d_model"])

        self.drop     = nn.Dropout(dropout)
        self.blocks   = nn.ModuleList([
            TransformerBlock(cfg["d_model"], cfg["n_heads"],
                             ffn_mult=4, max_seq=max_seq, dropout=dropout)
            for _ in range(cfg["n_layers"])
        ])
        self.ln_final = nn.LayerNorm(cfg["d_model"])

        # LM head: project back to vocabulary distribution
        # Weight tying: share weights between tok_emb and lm_head
        # This halves parameters and often improves perplexity
        self.lm_head = nn.Linear(cfg["d_model"], cfg["vocab_size"], bias=False)
        self.lm_head.weight = self.tok_emb.weight   # tie weights

        self._init_weights()

    def _init_weights(self):
        """GPT-2 style weight initialisation: N(0, 0.02), scaled residuals."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor,
                kv_caches: Optional[list] = None) -> Tuple[torch.Tensor, list]:
        """
        Forward pass.

        Args:
            input_ids : (B, T) integer token IDs
            kv_caches : list of (K, V) tuples per layer, or None for full attention

        Returns:
            logits    : (B, T, vocab_size) — raw scores over vocabulary
            new_caches: updated list of (K, V) per layer
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Token + positional embeddings
        # arange(T) creates [0, 1, ..., T-1] — position indices
        pos    = torch.arange(T, device=device).unsqueeze(0)   # (1, T)
        x      = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))

        # Pass through transformer blocks, threading KV caches
        new_caches = []
        for i, block in enumerate(self.blocks):
            cache_in = kv_caches[i] if kv_caches is not None else None
            x, cache_out = block(x, kv_cache=cache_in)
            new_caches.append(cache_out)

        x      = self.ln_final(x)
        logits = self.lm_head(x)   # (B, T, vocab_size)

        return logits, new_caches

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        """
        Autoregressive token generation with KV cache.

        TWO PHASES:
        1. Prefill: run full forward pass on prompt → populate KV cache
                    This is COMPUTE-BOUND (all tokens processed in parallel)
        2. Decode:  generate one token at a time using KV cache
                    This is MEMORY-BANDWIDTH BOUND (reads full KV cache each step)

        top_k sampling: keep only the k highest-probability tokens, sample from them.
        temperature > 1 = more random; < 1 = more deterministic; = 0 → greedy.
        """
        self.eval()
        device = input_ids.device

        # ── Phase 1: Prefill ─────────────────────────────────────────────────
        # Process entire prompt, build initial KV cache
        logits, kv_caches = self.forward(input_ids, kv_caches=None)

        for _ in range(max_new_tokens):
            # ── Phase 2: Decode (one token at a time) ────────────────────────
            # Only pass the LAST token (not the full sequence)
            # The KV cache holds all past context — this is the key insight
            last_logits = logits[:, -1, :]  # (B, vocab_size)

            # Apply temperature scaling
            if temperature > 0:
                last_logits = last_logits / temperature

            # Top-k filtering: zero out all but top-k logits
            if top_k > 0:
                v, _ = torch.topk(last_logits, min(top_k, last_logits.size(-1)))
                last_logits[last_logits < v[:, [-1]]] = float('-inf')

            # Sample next token
            probs     = F.softmax(last_logits, dim=-1)
            next_tok  = torch.multinomial(probs, num_samples=1)  # (B, 1)

            input_ids = torch.cat([input_ids, next_tok], dim=1)

            # Only forward the new token; KV cache handles the rest
            logits, kv_caches = self.forward(next_tok, kv_caches=kv_caches)

        return input_ids

    def param_count(self) -> str:
        """Return human-readable parameter count."""
        n = sum(p.numel() for p in self.parameters())
        return f"{n/1e6:.1f}M" if n >= 1e6 else f"{n/1e3:.1f}K"

    @staticmethod
    def estimate_memory_gb(size: str, batch: int = 1, seq: int = 512,
                           dtype_bytes: int = 2) -> dict:
        """
        Estimate GPU memory requirements for a given model and input size.
        Useful for planning before loading a model on constrained hardware.

        Components:
            weights     — model parameters × dtype_bytes
            activations — forward pass intermediate tensors (rough estimate)
            kv_cache    — key + value tensors per layer per token
            gradients   — same size as weights (training only)
        """
        cfg = TinyTransformer.CONFIGS[size]
        d   = cfg["d_model"]
        L   = cfg["n_layers"]
        V   = cfg["vocab_size"]
        H   = cfg["n_heads"]

        # Parameter count (rough):
        #   token emb + pos emb + L × (qkv + out + ffn×2 + 2×ln) + final_ln + lm_head
        params = V * d + 512 * d + L * (3*d*d + d*d + 2*4*d*d + 4*d) + d + V*d
        weight_gb = params * dtype_bytes / 1e9

        # KV cache per token per layer: 2 (K,V) × n_heads × head_dim × dtype_bytes
        head_dim = d // H
        kv_per_token = L * 2 * H * head_dim * dtype_bytes
        kv_cache_gb = batch * seq * kv_per_token / 1e9

        # Activations (very rough): ~6 × weight memory for training
        act_gb = weight_gb * 6

        return {
            "weights_gb":    round(weight_gb, 3),
            "kv_cache_gb":   round(kv_cache_gb, 3),
            "activations_gb": round(act_gb, 3),
            "total_train_gb": round(weight_gb * 4 + act_gb, 3),  # ×4 for optimizer states
            "total_infer_gb": round(weight_gb + kv_cache_gb, 3),
        }


# =============================================================================
# 2 — SMALL CNN (for DataLoader and basic training experiments)
# =============================================================================

class SmallCNN(nn.Module):
    """
    5-layer CNN for image classification.
    Lighter than ResNet — faster to train for DataLoader bottleneck experiments.
    Input: (B, C, 224, 224)   Output: (B, num_classes)
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32,  3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32,          64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64,          128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(128,         256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# =============================================================================
# 3 — MODEL FACTORY
# =============================================================================

def get_model(name: str, device: str = "cpu", **kwargs) -> nn.Module:
    """
    Factory function: return a model by name, moved to device.

    Usage:
        model = get_model("tiny_transformer", device="cuda", size="small")
        model = get_model("small_cnn",        device="cuda", num_classes=10)
        model = get_model("resnet18",          device="cuda")

    Supported names:
        tiny_transformer  — TinyTransformer with size= kwarg (tiny/small/medium/large)
        small_cnn         — SmallCNN
        resnet18/50       — torchvision ResNets
    """
    from torchvision import models as tv_models

    name_lower = name.lower()

    if name_lower == "tiny_transformer":
        size = kwargs.get("size", "small")
        max_seq = kwargs.get("max_seq", 512)
        model = TinyTransformer(size=size, max_seq=max_seq)

    elif name_lower == "small_cnn":
        model = SmallCNN(
            num_classes=kwargs.get("num_classes", 1000)
        )

    elif name_lower == "resnet18":
        model = tv_models.resnet18(weights=None)
        if "num_classes" in kwargs and kwargs["num_classes"] != 1000:
            model.fc = nn.Linear(512, kwargs["num_classes"])

    elif name_lower == "resnet50":
        model = tv_models.resnet50(weights=None)
        if "num_classes" in kwargs and kwargs["num_classes"] != 1000:
            model.fc = nn.Linear(2048, kwargs["num_classes"])

    else:
        raise ValueError(f"Unknown model: {name}. Options: tiny_transformer, small_cnn, resnet18, resnet50")

    model = model.to(device)
    return model


# =============================================================================
# 4 — QUICK SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("model.py — Self Test\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # ── Test TinyTransformer ─────────────────────────────────────────────────
    for size in ["tiny", "small", "medium"]:
        m = TinyTransformer(size=size)
        dummy = torch.randint(0, 50257, (2, 64))   # batch=2, seq=64
        logits, caches = m(dummy)
        mem = TinyTransformer.estimate_memory_gb(size)
        print(f"  TinyTransformer [{size:6s}]: {m.param_count():>8} params | "
              f"logits={tuple(logits.shape)} | "
              f"infer_mem={mem['total_infer_gb']:.3f}GB | "
              f"train_mem={mem['total_train_gb']:.3f}GB")

    print()

    # ── Test generation with KV cache ─────────────────────────────────────
    m = TinyTransformer(size="tiny").to(device)
    prompt = torch.randint(0, 50257, (1, 10), device=device)
    output = m.generate(prompt, max_new_tokens=20, temperature=0.8)
    print(f"  Generation test: prompt={prompt.shape} → output={output.shape} ✓")

    # ── Test SmallCNN ────────────────────────────────────────────────────────
    cnn = SmallCNN().to(device)
    imgs = torch.rand(4, 3, 224, 224, device=device)
    out  = cnn(imgs)
    print(f"  SmallCNN: input={tuple(imgs.shape)} → output={tuple(out.shape)} ✓")

    # ── Test factory ─────────────────────────────────────────────────────────
    m2 = get_model("tiny_transformer", device=device, size="tiny")
    print(f"  get_model factory: {type(m2).__name__} on {device} ✓")

    print("\nAll tests passed.")
