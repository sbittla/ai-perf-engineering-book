#!/usr/bin/env bash
# =============================================================================
# vllm_benchmark.sh  —  Project 1, Step 6: vLLM Throughput + Latency Benchmarks
# =============================================================================
# PURPOSE:
#   Run vLLM's official benchmark scripts against a running vLLM server
#   to measure throughput (tokens/sec) and latency (TTFT, TBT, P99).
#
# PRE-REQUISITE:
#   vLLM server must be running. Start it with:
#     bash vllm_server_start.sh gpt2 baseline    (terminal 1)
#     bash vllm_benchmark.sh                     (terminal 2)
#
# METRICS MEASURED:
#   - Throughput    : total tokens generated per second (all requests combined)
#   - TTFT          : Time To First Token — how long before user sees first word
#   - TBT           : Time Between Tokens — per-token decode latency
#   - P50/P99       : latency percentiles — P99 matters for user experience
#   - Request RPS   : how many requests/second the server can sustain
#
# UNDERSTANDING TTFT vs THROUGHPUT:
#   High throughput ≠ low latency.
#   vLLM batches requests together for throughput. This increases TTFT
#   for individual requests because it waits to fill a batch.
#   In production you tune the tradeoff based on your SLA.
# =============================================================================

set -euo pipefail

MODEL="${1:-gpt2}"
HOST="${2:-localhost}"
PORT="${3:-8000}"
VLLM_DIR="${VLLM_DIR:-$(pip show vllm 2>/dev/null | grep Location | awk '{print $2}')/vllm}"
OUTDIR="benchmark_results"
mkdir -p "$OUTDIR"

echo "============================================================"
echo "  vLLM Benchmarks"
echo "  Model: $MODEL   Host: $HOST:$PORT"
echo "============================================================"

# ── Verify server is running ─────────────────────────────────────────────────
echo ""
echo "Checking server health..."
curl -s "http://$HOST:$PORT/health" > /dev/null 2>&1 || {
    echo "ERROR: vLLM server not responding at $HOST:$PORT"
    echo "Start it first: bash vllm_server_start.sh $MODEL baseline"
    exit 1
}
echo "  ✓ Server is up"

# ── Benchmark 1: Offline Throughput ─────────────────────────────────────────
echo ""
echo "[1/4] Offline Throughput Benchmark..."
echo "      Sends all prompts at once, measures total tokens/second"
echo "      (Best case for GPU utilisation — no request rate limiting)"
echo ""

# --num-prompts  : total number of prompts to process
# --input-len    : number of input tokens per prompt (fixed synthetic prompts)
# --output-len   : number of output tokens to generate per prompt
# Offline mode = submit all at once, measure total time, compute tok/s
python3 -c "
# Inline version using vLLM's LLM class directly (no server needed)
from vllm import LLM, SamplingParams
import time, torch

print('Loading model directly via vLLM LLM class...')
llm = LLM(model='$MODEL', dtype='float16')

# SamplingParams controls decoding strategy:
#   temperature=0 = greedy (deterministic, best for benchmarking)
#   max_tokens    = how many tokens to generate per prompt
params = SamplingParams(temperature=0, max_tokens=100)

# Create 50 synthetic prompts
prompts = ['The quick brown fox jumps over the lazy dog.'] * 50

# Warmup: run once to trigger CUDA compilation
_ = llm.generate(prompts[:2], params)
torch.cuda.synchronize()

# Timed run
start = time.perf_counter()
outputs = llm.generate(prompts, params)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
tps = total_tokens / elapsed

print(f'')
print(f'  Prompts    : {len(prompts)}')
print(f'  Total time : {elapsed:.2f}s')
print(f'  Total tok  : {total_tokens}')
print(f'  Throughput : {tps:.1f} tok/s')
print(f'  Avg lat    : {elapsed/len(prompts)*1000:.1f} ms/request')
" 2>&1 | tee "$OUTDIR/throughput_offline.txt"

# ── Benchmark 2: Latency at batch=1 ─────────────────────────────────────────
echo ""
echo "[2/4] Single-Request Latency Benchmark..."
echo "      One request at a time — measures minimum achievable latency"
echo "      This is the baseline for interactive/chat use cases"
echo ""

python3 -c "
from vllm import LLM, SamplingParams
import time, torch, statistics

llm = LLM(model='$MODEL', dtype='float16')
params = SamplingParams(temperature=0, max_tokens=50)
prompt = ['Explain the concept of neural networks in simple terms:']

# Warmup
_ = llm.generate(prompt, params)

latencies = []
for i in range(10):
    start = time.perf_counter()
    out = llm.generate(prompt, params)
    torch.cuda.synchronize()
    latencies.append((time.perf_counter() - start) * 1000)

tokens_out = len(out[0].outputs[0].token_ids)
print(f'  Output tokens  : {tokens_out}')
print(f'  Latency P50    : {statistics.median(latencies):.1f} ms')
print(f'  Latency P95    : {sorted(latencies)[int(0.95*len(latencies))]:.1f} ms')
print(f'  Latency P99    : {sorted(latencies)[-1]:.1f} ms')
print(f'  Tok/s (single) : {tokens_out / (statistics.mean(latencies)/1000):.1f}')
" 2>&1 | tee "$OUTDIR/latency_single.txt"

# ── Benchmark 3: Variable-length prompts (realistic workload) ────────────────
echo ""
echo "[3/4] Realistic Variable-Length Benchmark..."
echo "      Mix of short (128 tok) and long (512 tok) prompts"
echo "      Simulates real chat traffic better than fixed lengths"
echo ""

python3 -c "
from vllm import LLM, SamplingParams
import time, torch, random

llm = LLM(model='$MODEL', dtype='float16')

# Variable output lengths simulate real user requests
prompts_and_params = [
    ('Tell me a short story about AI:', SamplingParams(temperature=0.7, max_tokens=200)),
    ('What is 2+2?', SamplingParams(temperature=0, max_tokens=10)),
    ('Summarise the history of computing:', SamplingParams(temperature=0, max_tokens=150)),
    ('Write a haiku about GPUs:', SamplingParams(temperature=0.8, max_tokens=30)),
] * 10  # Repeat to get 40 requests

prompts = [p for p,_ in prompts_and_params]
params  = [p for _,p in prompts_and_params]

start = time.perf_counter()
outputs = llm.generate(prompts, params)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
print(f'  Requests   : {len(prompts)}')
print(f'  Total time : {elapsed:.2f}s')
print(f'  Total tok  : {total_tokens}')
print(f'  Throughput : {total_tokens/elapsed:.1f} tok/s')
" 2>&1 | tee "$OUTDIR/throughput_variable.txt"

# ── Benchmark 4: Serving benchmark against live server ───────────────────────
echo ""
echo "[4/4] Serving Benchmark (if server is running on port $PORT)..."
echo "      Simulates N requests/second hitting the live server"
echo ""

# Check if vllm benchmark_serving.py exists
BENCH_SERVING=$(find / -name "benchmark_serving.py" -path "*/vllm/*" 2>/dev/null | head -1 || echo "")

if [ -n "$BENCH_SERVING" ]; then
    # --request-rate : requests per second (use inf for max throughput mode)
    # --num-prompts  : total requests to send before stopping
    python3 "$BENCH_SERVING" \
        --backend vllm \
        --host "$HOST" \
        --port "$PORT" \
        --model "$MODEL" \
        --num-prompts 100 \
        --request-rate 5 \
        2>&1 | tee "$OUTDIR/serving_benchmark.txt"
else
    echo "  benchmark_serving.py not found — run manually:"
    echo "  python3 \$(python3 -c 'import vllm; import os; print(os.path.dirname(vllm.__file__))')/benchmarks/benchmark_serving.py \\"
    echo "    --backend vllm --host $HOST --port $PORT --model $MODEL \\"
    echo "    --num-prompts 100 --request-rate 5"
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  BENCHMARK SUMMARY"
echo "  Results saved in: $OUTDIR/"
ls -1 "$OUTDIR/"
echo ""
echo "  NEXT STEPS:"
echo "  1. Note throughput with baseline config"
echo "  2. Restart server with 'prefix' config: bash vllm_server_start.sh $MODEL prefix"
echo "  3. Re-run benchmarks and compare"
echo "  4. Run torch_compile_bench.py for torch.compile comparison"
echo "============================================================"
