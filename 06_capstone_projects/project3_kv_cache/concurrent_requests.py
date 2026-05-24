#!/usr/bin/env python3
"""
concurrent_requests.py  —  Project 3: KV Cache Pressure via Concurrent Load
=============================================================================

HOW TO RUN:
    # Terminal 1: Start vLLM server (constrained for experiment)
    bash kv_pressure_server.sh

    # Terminal 2: Monitor GPU memory
    bash memory_monitor.sh &

    # Terminal 3: Run this benchmark
    python concurrent_requests.py --concurrency 1   # baseline
    python concurrent_requests.py --concurrency 5   # moderate
    python concurrent_requests.py --concurrency 20  # stress test


"""

import argparse
import asyncio
import time
import statistics
import json
import sys
from dataclasses import dataclass, field
from typing import List

try:
    import aiohttp
except ImportError:
    print("Install: pip install aiohttp")
    sys.exit(1)

parser = argparse.ArgumentParser()
parser.add_argument("--host",        default="localhost")
parser.add_argument("--port",        type=int, default=8000)
parser.add_argument("--concurrency", type=int, default=5,
                    help="Number of simultaneous requests")
parser.add_argument("--total",       type=int, default=50,
                    help="Total requests to send")
parser.add_argument("--input-len",   type=int, default=256,
                    help="Number of input tokens (approximate via word count)")
parser.add_argument("--output-len",  type=int, default=200,
                    help="Number of output tokens to generate")
parser.add_argument("--timeout",     type=float, default=120.0,
                    help="Request timeout in seconds")
args = parser.parse_args()

BASE_URL = f"http://{args.host}:{args.port}"

# ── Test prompts of varying lengths ──────────────────────────────────────────
# Repeat words to approximate desired token count
# Rough rule: 1 token ≈ 0.75 words in English

def make_prompt(approx_tokens: int) -> str:
    """Create a prompt of approximately the specified token length."""
    base = ("The history of artificial intelligence is a fascinating story "
            "of human ambition and technical achievement. Researchers have ")
    # Repeat to hit approximate token count (very rough)
    word_count = int(approx_tokens * 0.75)
    words = base.split()
    repeated = (words * (word_count // len(words) + 1))[:word_count]
    prompt = " ".join(repeated)
    prompt += ". Please continue this discussion by explaining:"
    return prompt

SHORT_PROMPT = make_prompt(64)
MEDIUM_PROMPT = make_prompt(args.input_len)
LONG_PROMPT   = make_prompt(min(args.input_len * 2, 1024))

# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class RequestResult:
    request_id:   int
    success:      bool
    ttft_s:       float = 0.0       # Time to first token (seconds)
    total_s:      float = 0.0       # Total request duration
    output_tokens:int   = 0
    error:        str   = ""

# ── Single request coroutine ──────────────────────────────────────────────────
async def send_request(
    session: aiohttp.ClientSession,
    request_id: int,
    prompt: str,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> RequestResult:
    """
    Send one request to the vLLM OpenAI-compatible endpoint.
    Uses streaming (stream=True) so we can measure TTFT precisely.
    
    STREAMING vs NON-STREAMING:
        Non-streaming: server generates ALL tokens, then sends one HTTP response.
                       You can't measure TTFT.
        Streaming:     server sends each token as a Server-Sent Event (SSE).
                       We record time of first event = TTFT.
    """
    # Semaphore limits how many requests run concurrently
    # This is how we control concurrency without launching too many coroutines
    async with semaphore:
        start = time.perf_counter()
        first_token_time = None
        output_tokens = 0
        
        payload = {
            "model": "gpt2",         # Must match the model loaded in the server
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,        # Greedy decoding — deterministic
            "stream": True,          # Enable streaming for TTFT measurement
        }
        
        try:
            async with session.post(
                f"{BASE_URL}/v1/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=args.timeout),
            ) as response:
                
                if response.status != 200:
                    return RequestResult(
                        request_id=request_id,
                        success=False,
                        error=f"HTTP {response.status}"
                    )
                
                # Read streaming response (Server-Sent Events format)
                # Each line is "data: {json}" or "data: [DONE]"
                async for line in response.content:
                    line_str = line.decode("utf-8").strip()
                    
                    if not line_str or not line_str.startswith("data:"):
                        continue
                    
                    data_str = line_str[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        chunk = json.loads(data_str)
                        token_text = chunk.get("choices", [{}])[0].get("text", "")
                        
                        if token_text and first_token_time is None:
                            # First non-empty token = TTFT
                            first_token_time = time.perf_counter()
                        
                        if token_text:
                            output_tokens += 1
                    except json.JSONDecodeError:
                        continue
                
                total_s = time.perf_counter() - start
                ttft_s  = (first_token_time - start) if first_token_time else total_s
                
                return RequestResult(
                    request_id=request_id,
                    success=True,
                    ttft_s=ttft_s,
                    total_s=total_s,
                    output_tokens=output_tokens,
                )
        
        except asyncio.TimeoutError:
            return RequestResult(request_id=request_id, success=False, error="TIMEOUT")
        except Exception as e:
            return RequestResult(request_id=request_id, success=False, error=str(e))

# ── Main benchmark runner ─────────────────────────────────────────────────────
async def run_benchmark(concurrency: int, total_requests: int) -> List[RequestResult]:
    """Run the full benchmark with given concurrency level."""
    
    semaphore = asyncio.Semaphore(concurrency)
    
    # aiohttp connector: limit total connections to avoid overwhelming server
    connector = aiohttp.TCPConnector(limit=concurrency + 5)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        # Create all request tasks
        tasks = []
        for i in range(total_requests):
            # Alternate between short and medium prompts for variety
            prompt = MEDIUM_PROMPT if i % 3 != 0 else SHORT_PROMPT
            task = asyncio.create_task(
                send_request(session, i, prompt, args.output_len, semaphore)
            )
            tasks.append(task)
        
        # Run with progress display
        results = []
        completed = 0
        
        start_time = time.perf_counter()
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1
            
            # Live progress every 5 requests
            if completed % 5 == 0 or completed == total_requests:
                elapsed = time.perf_counter() - start_time
                rps = completed / elapsed
                success_count = sum(1 for r in results if r.success)
                print(f"  Progress: {completed}/{total_requests}  "
                      f"RPS={rps:.1f}  "
                      f"Success={success_count}/{completed}", end="\r")
        
        print()  # newline after progress
        return results

# ── Analyse and print results ─────────────────────────────────────────────────
def print_results(results: List[RequestResult], concurrency: int, elapsed_s: float):
    successful = [r for r in results if r.success]
    failed     = [r for r in results if not r.success]
    
    if not successful:
        print(f"  ERROR: All {len(results)} requests failed!")
        for r in failed[:3]:
            print(f"    Request {r.request_id}: {r.error}")
        return
    
    ttft_vals   = [r.ttft_s  * 1000 for r in successful]  # convert to ms
    total_vals  = [r.total_s * 1000 for r in successful]
    tps_vals    = [r.output_tokens / r.total_s for r in successful if r.total_s > 0]
    
    sorted_ttft  = sorted(ttft_vals)
    sorted_total = sorted(total_vals)
    
    total_tokens = sum(r.output_tokens for r in successful)
    overall_tps  = total_tokens / elapsed_s
    
    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │  CONCURRENCY = {concurrency:<3}  Results                  │")
    print(f"  ├─────────────────────────────────────────────┤")
    print(f"  │  Total requests   : {len(results):<24} │")
    print(f"  │  Successful       : {len(successful):<24} │")
    print(f"  │  Failed/Timeout   : {len(failed):<24} │")
    print(f"  │  Elapsed          : {elapsed_s:.1f}s{'':<22} │")
    print(f"  │  Throughput       : {overall_tps:.1f} tok/s{'':<19} │")
    print(f"  │  RPS sustained    : {len(successful)/elapsed_s:.1f} req/s{'':<20} │")
    print(f"  ├─────────────────────────────────────────────┤")
    print(f"  │  TTFT  P50        : {sorted_ttft[len(sorted_ttft)//2]:.0f}ms{'':<23} │")
    print(f"  │  TTFT  P95        : {sorted_ttft[int(0.95*len(sorted_ttft))]:.0f}ms{'':<23} │")
    print(f"  │  TTFT  P99        : {sorted_ttft[-1]:.0f}ms{'':<23} │")
    print(f"  │  Total P50        : {sorted_total[len(sorted_total)//2]:.0f}ms{'':<23} │")
    print(f"  │  Total P99        : {sorted_total[-1]:.0f}ms{'':<23} │")
    print(f"  └─────────────────────────────────────────────┘")
    
    if failed:
        print(f"\n  Failed request errors:")
        for r in failed[:5]:
            print(f"    [{r.request_id}] {r.error}")

# ── Entry point ───────────────────────────────────────────────────────────────
async def main():
    print(f"\n{'='*55}")
    print(f"  KV Cache Pressure Benchmark")
    print(f"  Server: {BASE_URL}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Total requests: {args.total}")
    print(f"  Input tokens (approx): {args.input_len}")
    print(f"  Output tokens: {args.output_len}")
    print(f"{'='*55}")
    
    # Check server health first
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{BASE_URL}/health", timeout=aiohttp.ClientTimeout(total=5)) as r:
                if r.status == 200:
                    print(f"\n  ✓ Server is up at {BASE_URL}")
    except Exception:
        print(f"\n  ERROR: Cannot reach server at {BASE_URL}")
        print(f"  Start it with: bash kv_pressure_server.sh")
        return
    
    print(f"\n  Starting benchmark (watch GPU memory in another terminal):")
    print(f"    nvidia-smi dmon -s mu -d 1\n")
    
    start = time.perf_counter()
    results = await run_benchmark(args.concurrency, args.total)
    elapsed = time.perf_counter() - start
    
    print_results(results, args.concurrency, elapsed)

if __name__ == "__main__":
    asyncio.run(main())
