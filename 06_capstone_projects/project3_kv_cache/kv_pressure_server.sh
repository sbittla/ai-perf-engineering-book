#!/usr/bin/env bash
# =============================================================================
# kv_pressure_server.sh  —  Project 3: Start vLLM Server Configurations
# =============================================================================
# PURPOSE:
#   Start the vLLM server with configurations that intentionally expose
#   KV cache limits so you can observe and then fix the behaviour.
#
# EXPERIMENT FLOW:
#   Run each config, then hit it with concurrent_requests.py at high concurrency.
#   Watch memory_monitor.sh in another terminal throughout.
#
#   Config 1 (constrained)  → memory fills → requests queue / OOM
#   Config 2 (prefix cache) → shared prefixes reuse cache → more headroom
#   Config 3 (quantized)    → smaller weights → more space for KV cache
#   Config 4 (chunked)      → long prefills split → better latency fairness
# =============================================================================

MODEL="${1:-gpt2}"
CONFIG="${2:-constrained}"

echo "============================================================"
echo "  vLLM KV Cache Experiment Server"
echo "  Config: $CONFIG"
echo "============================================================"
echo ""
echo "In another terminal, start memory monitoring:"
echo "  bash memory_monitor.sh"
echo ""
echo "Then send load:"
echo "  python concurrent_requests.py --concurrency 20 --total 100"
echo ""

case "$CONFIG" in

  # ── Config 1: Constrained memory — triggers KV cache pressure ────────────
  constrained)
    echo "CONFIG: Low gpu-memory-utilization (0.6) — forces KV cache pressure"
    echo ""
    echo "WHY:"
    echo "  gpu_memory_utilization=0.6 means only 60% of VRAM goes to KV cache."
    echo "  Under 20 concurrent requests, this will fill up quickly."
    echo "  vLLM will start evicting KV blocks, causing recomputation."
    echo "  You'll see: latency spikes, memory hovering at 60% of total VRAM."
    echo ""
    python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" \
      --dtype float16 \
      --port 8000 \
      --max-model-len 2048 \
      --gpu-memory-utilization 0.60 \
      --max-num-seqs 8 \
      --disable-log-requests
    ;;

  # ── Config 2: Optimised baseline ─────────────────────────────────────────
  optimized)
    echo "CONFIG: Higher memory utilization (0.90) — more KV cache headroom"
    echo ""
    echo "WHY:"
    echo "  More VRAM goes to KV cache → more concurrent requests supported."
    echo "  Compare P99 latency and RPS to 'constrained' config."
    echo ""
    python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" \
      --dtype float16 \
      --port 8000 \
      --max-model-len 2048 \
      --gpu-memory-utilization 0.90 \
      --max-num-seqs 32 \
      --disable-log-requests
    ;;

  # ── Config 3: Prefix caching ──────────────────────────────────────────────
  prefix)
    echo "CONFIG: Prefix Caching enabled"
    echo ""
    echo "WHY PREFIX CACHING HELPS:"
    echo "  If many requests share the same system prompt (e.g. 'You are a"
    echo "  helpful assistant. Always respond in English.'), vLLM computes"
    echo "  the KV cache for this prefix ONCE and shares it across requests."
    echo ""
    echo "  BENEFIT: Reduces TTFT for requests with shared prefixes."
    echo "  BENEFIT: Reduces GPU compute (no redundant prefill)."
    echo "  COST:    Slightly more memory management overhead."
    echo ""
    echo "  TEST: Send requests with the same long system prompt and observe"
    echo "  that TTFT decreases after the first request (cache hit)."
    echo ""
    python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" \
      --dtype float16 \
      --port 8001 \
      --max-model-len 2048 \
      --gpu-memory-utilization 0.85 \
      --enable-prefix-caching \
      --disable-log-requests
    ;;

  # ── Config 4: Chunked prefill ─────────────────────────────────────────────
  chunked)
    echo "CONFIG: Chunked Prefill"
    echo ""
    echo "WHY:"
    echo "  A long prompt (e.g. 1024 tokens) without chunking monopolises the"
    echo "  GPU for its entire prefill phase. Other requests must wait."
    echo "  Chunked prefill breaks long prefills into chunks of N tokens,"
    echo "  interleaving them with decode steps from other requests."
    echo ""
    echo "  EFFECT: Lower P99 TTFT under mixed workloads."
    echo "  TRADEOFF: Slightly lower total throughput (interleaving overhead)."
    echo ""
    python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" \
      --dtype float16 \
      --port 8002 \
      --max-model-len 2048 \
      --gpu-memory-utilization 0.85 \
      --enable-chunked-prefill \
      --max-num-batched-tokens 512 \
      --disable-log-requests
    ;;

  *)
    echo "Unknown config: $CONFIG"
    echo "Available: constrained | optimized | prefix | chunked"
    exit 1
    ;;
esac
