#!/usr/bin/env bash
# =============================================================================
# vllm_server_start.sh  —  Project 1, Step 5: Start vLLM Inference Server
# =============================================================================
# PURPOSE:
#   Start a vLLM OpenAI-compatible inference server with various configurations
#   so you can benchmark throughput and latency via HTTP requests.
#
# WHAT IS vLLM?
#   vLLM implements PagedAttention — a virtual memory system for the KV cache.
#   Instead of pre-allocating a fixed KV cache per sequence (which wastes memory
#   on sequences shorter than the max), vLLM allocates KV cache in pages on demand.
#   This allows much higher concurrency: more requests in flight simultaneously,
#   which improves GPU utilisation and throughput.
#
# HOW TO USE:
#   # Terminal 1: Start server
#   bash vllm_server_start.sh gpt2
#
#   # Terminal 2: Send requests
#   bash vllm_benchmark.sh
#
# ARGUMENTS:
#   $1 = model name (default: gpt2 for quick testing)
#   $2 = config preset: "baseline" | "fp16" | "quantized" | "prefix"
# =============================================================================

MODEL="${1:-gpt2}"
CONFIG="${2:-baseline}"

echo "============================================================"
echo "  Starting vLLM Server"
echo "  Model  : $MODEL"
echo "  Config : $CONFIG"
echo "============================================================"
echo ""

case "$CONFIG" in

  # ── Default: FP16, single GPU ─────────────────────────────────────────────
  baseline)
    echo "Config: FP16, no special optimisations (comparable to HuggingFace)"
    python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" \
      --dtype float16 \
      --port 8000 \
      --max-model-len 2048 \
      --gpu-memory-utilization 0.85
      # --gpu-memory-utilization: fraction of GPU VRAM reserved for KV cache
      # 0.85 = leave 15% for model weights and activations
    ;;

  # ── FP16 with prefix caching ──────────────────────────────────────────────
  prefix)
    echo "Config: FP16 + Prefix Caching (reuses KV cache for identical prefixes)"
    echo ""
    echo "WHY PREFIX CACHING:"
    echo "  If many requests share the same system prompt, vLLM can compute"
    echo "  the KV cache for that prefix ONCE and reuse it. This reduces:"
    echo "  - Time-to-first-token for subsequent requests"
    echo "  - GPU compute spent on redundant prefill"
    echo ""
    python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" \
      --dtype float16 \
      --port 8001 \
      --enable-prefix-caching \
      --gpu-memory-utilization 0.85
    ;;

  # ── Chunked prefill ───────────────────────────────────────────────────────
  chunked)
    echo "Config: FP16 + Chunked Prefill"
    echo ""
    echo "WHY CHUNKED PREFILL:"
    echo "  Long prompts (2048+ tokens) can monopolise the GPU during prefill,"
    echo "  causing high TTFT for other requests. Chunked prefill breaks the"
    echo "  prefill into smaller chunks, interleaving with decode steps."
    echo "  This reduces latency for concurrent users at slight throughput cost."
    echo ""
    python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" \
      --dtype float16 \
      --port 8002 \
      --enable-chunked-prefill \
      --max-num-batched-tokens 2048 \
      --gpu-memory-utilization 0.85
    ;;

  # ── Quantized (INT8 or AWQ) ───────────────────────────────────────────────
  quantized)
    echo "Config: INT8/AWQ Quantization"
    echo ""
    echo "WHY QUANTIZE:"
    echo "  LLM inference on RTX 4060 (8GB) is often memory-limited."
    echo "  INT8 weights use half the memory of FP16, allowing:"
    echo "  - Larger models to fit in 8GB VRAM"
    echo "  - Larger KV cache (more concurrent requests)"
    echo "  - Potentially higher throughput if memory bandwidth was the limit"
    echo ""
    # AWQ (Activation-aware Weight Quantization) preserves accuracy better than
    # naive INT8 by protecting the most important weights from quantization error
    python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" \
      --dtype float16 \
      --quantization awq \
      --port 8003 \
      --gpu-memory-utilization 0.90
    ;;

  *)
    echo "Unknown config: $CONFIG"
    echo "Options: baseline | prefix | chunked | quantized"
    exit 1
    ;;
esac
