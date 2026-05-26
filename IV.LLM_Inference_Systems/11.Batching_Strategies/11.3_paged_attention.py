#!/usr/bin/env python3
"""
IV.LLM_Inference_Systems/11.Batching_Strategies/11.3_paged_attention.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chapter 11: Batching Strategies — Section 3: PagedAttention
=======================================================================
Covers book section 11.3:
  • Static allocation: pre-reserves max_seq_len KV memory per request
  • Internal fragmentation: reserved memory never used by short responses
  • PagedAttention: divides KV cache into fixed-size blocks (pages)
  • Block allocator: allocate on demand, free on completion
  • How PagedAttention improves concurrent request count

Run:  python IV.LLM_Inference_Systems/11.Batching_Strategies/11.3_paged_attention.py
All sections must print ✓.
"""

import math
import random

print("=" * 60)
print("  Exercise 11.3 — PagedAttention and Block Allocation")
print("=" * 60)
print()


# ─────────────────────────────────────────────────────────────
# SECTION 1: The KV cache fragmentation problem
# ─────────────────────────────────────────────────────────────
print("── Section 1: Static KV Allocation and Fragmentation ──")
print("""
  NAIVE APPROACH (HuggingFace generate):
    When a request arrives, pre-allocate a contiguous block of KV memory
    sized for max_seq_len tokens, regardless of how many tokens will
    actually be generated.

  PROBLEM — INTERNAL FRAGMENTATION:
    If max_seq_len = 2048 and the request generates only 100 tokens:
      Reserved  : 2048 × kv_bytes_per_token
      Used      : 100  × kv_bytes_per_token
      Wasted    : 1948 × kv_bytes_per_token = 95.1% waste!

    In a fleet with varied output lengths, static allocation means
    most of the KV pool is reserved but unused — fewer concurrent
    requests fit in VRAM.

  PROBLEM — EXTERNAL FRAGMENTATION:
    When requests complete, their blocks are freed. New requests need
    contiguous blocks of max_seq_len. The free space may be scattered
    across many small gaps — unusable for new large requests.
""")
print("  ✓ Section 1 passed — static allocation causes severe fragmentation")


# ─────────────────────────────────────────────────────────────
# SECTION 2: PagedAttention block allocator
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: PagedAttention Block Allocator ──")
print("""
  PagedAttention (vLLM) treats KV memory like virtual memory pages.
  The entire KV pool is divided into fixed-size blocks.
  Each block holds BLOCK_SIZE tokens worth of K and V per layer.

  On each decode step, if a sequence needs more KV space:
    → Allocate one more block from the free pool
    → Map it into the sequence's "page table"

  When a sequence finishes:
    → All its blocks are returned to the free pool immediately
    → No compaction needed (non-contiguous access is fine for GPU)

  BLOCK_SIZE trade-off:
    Small blocks (e.g. 4):  low waste, many bookkeeping entries
    Large blocks (e.g. 64): fewer entries, up to 63 tokens wasted per request

  vLLM default: --block-size 16 (16 tokens per block)

  TODO 1: Implement PagedAllocator class below.
""")


class PagedAllocator:
    """
    TODO 1: Implement a block-based KV cache allocator.

    Attributes:
      total_blocks  : total number of blocks in the pool
      free_blocks   : set of block IDs currently free
      allocations   : dict mapping request_id → list of block IDs

    Methods:
      allocate(req_id, n_blocks) : allocate n_blocks for req_id
                                   raise RuntimeError if not enough free
      free(req_id)               : return all blocks of req_id to free pool
      blocks_used()              : total allocated blocks
      blocks_free()              : total free blocks
    """

    def __init__(self, total_blocks: int):
        self.total_blocks = total_blocks
        self.free_blocks  = set(range(total_blocks))
        self.allocations  = {}   # req_id → [block_id, ...]

    def allocate(self, req_id: int, n_blocks: int) -> list:
        """
        TODO 1a: Allocate n_blocks for req_id.
        Pop n_blocks from free_blocks (raise RuntimeError if not enough).
        Store in self.allocations[req_id].
        Return the list of allocated block IDs.
        """
        # YOUR CODE HERE
        if len(self.free_blocks) < n_blocks:
            raise RuntimeError(
                f"Out of KV blocks: need {n_blocks}, have {len(self.free_blocks)}"
            )
        chosen = []
        for _ in range(n_blocks):
            chosen.append(self.free_blocks.pop())
        if req_id in self.allocations:
            self.allocations[req_id].extend(chosen)
        else:
            self.allocations[req_id] = chosen
        return chosen

    def free(self, req_id: int):
        """
        TODO 1b: Return all blocks belonging to req_id to the free pool.
        """
        # YOUR CODE HERE
        if req_id in self.allocations:
            for blk in self.allocations.pop(req_id):
                self.free_blocks.add(blk)

    def blocks_used(self) -> int:
        return sum(len(v) for v in self.allocations.values())

    def blocks_free(self) -> int:
        return len(self.free_blocks)


# Test the allocator
alloc = PagedAllocator(total_blocks=100)
assert alloc.blocks_free() == 100, "Should start with all blocks free"

blks0 = alloc.allocate(req_id=0, n_blocks=10)
assert len(blks0) == 10
assert alloc.blocks_free() == 90
assert alloc.blocks_used() == 10

blks1 = alloc.allocate(req_id=1, n_blocks=20)
assert alloc.blocks_free() == 70

alloc.free(req_id=0)
assert alloc.blocks_free() == 80, f"Expected 80 free, got {alloc.blocks_free()}"
assert 0 not in alloc.allocations

# Test OOM
alloc2 = PagedAllocator(total_blocks=5)
try:
    alloc2.allocate(req_id=0, n_blocks=10)
    assert False, "Should have raised RuntimeError"
except RuntimeError:
    pass

print(f"  PagedAllocator: allocate/free/OOM all working")
print("  ✓ Section 2 passed — block allocator implemented")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Static vs paged — concurrent request count
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: How Many Requests Fit? ──")
print("""
  We compare how many concurrent requests fit in the KV pool
  under static (max_seq_len) vs paged (actual tokens used) allocation.
""")

BLOCK_SIZE   = 16     # tokens per block (vLLM default)
MAX_SEQ_LEN  = 2048   # tokens (static allocation max)

# KV bytes per token for Llama-7B FP16:  2 * 32 * 32 * 128 * 2 = 524,288 bytes
KV_PER_TOKEN = 524_288   # bytes
KV_PER_BLOCK = KV_PER_TOKEN * BLOCK_SIZE   # bytes per block
KV_POOL_GB   = 8.0       # GB of VRAM reserved for KV cache

total_blocks = int(KV_POOL_GB * 1e9 / KV_PER_BLOCK)

print(f"  Model: Llama-7B FP16")
print(f"  KV pool: {KV_POOL_GB:.0f} GB")
print(f"  Block size: {BLOCK_SIZE} tokens")
print(f"  KV per block: {KV_PER_BLOCK/1e6:.1f} MB")
print(f"  Total blocks: {total_blocks}")

# Static allocation: each request reserves blocks for MAX_SEQ_LEN
blocks_per_req_static = math.ceil(MAX_SEQ_LEN / BLOCK_SIZE)
n_concurrent_static   = total_blocks // blocks_per_req_static

# PagedAttention: requests use only actual tokens generated
# Real output lengths drawn from a realistic distribution
random.seed(42)
N_REQUESTS = 200
output_lengths = [
    random.choices([50, 100, 200, 500, 1000],
                   weights=[0.30, 0.30, 0.25, 0.10, 0.05])[0]
    for _ in range(N_REQUESTS)
]
avg_actual_tokens = sum(output_lengths) / len(output_lengths)
blocks_per_req_paged = math.ceil(avg_actual_tokens / BLOCK_SIZE)
n_concurrent_paged   = total_blocks // blocks_per_req_paged

print(f"\n  Average actual output length: {avg_actual_tokens:.0f} tokens")
print(f"\n  {'Method':<25}  {'Blocks/req':>12}  {'Max concurrent':>15}")
print(f"  {'─'*25}  {'─'*12}  {'─'*15}")
print(f"  {'Static (max_seq_len)':<25}  {blocks_per_req_static:>12}  {n_concurrent_static:>15}")
print(f"  {'PagedAttention (actual)':<25}  {blocks_per_req_paged:>12}  {n_concurrent_paged:>15}")
print(f"  PagedAttention serves {n_concurrent_paged/max(n_concurrent_static,1):.1f}× more requests")

assert n_concurrent_paged > n_concurrent_static, \
    f"PagedAttention should serve more: {n_concurrent_paged} vs {n_concurrent_static}"
print("  ✓ Section 3 passed — PagedAttention dramatically increases concurrency")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Simulating the block allocator under load
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Block Allocator Under Continuous Load ──")
print("""
  We simulate a serving loop where requests arrive, consume blocks
  one per decode step, then release blocks on completion.

  This mirrors vLLM's internal block manager — requests get blocks
  as they decode and return them immediately when done.
""")


def simulate_paged(
    requests: list,      # list of output_lengths
    max_concurrency: int,
    total_blocks: int,
    block_size: int,
    decode_ms_per_step: float,
) -> dict:
    """
    Simulate continuous batching with PagedAttention block allocation.
    Returns throughput and peak block utilisation.
    """
    alloc   = PagedAllocator(total_blocks)
    waiting = list(requests)
    active  = {}   # req_id → steps_remaining
    done    = 0
    total_tokens = 0
    t_ms    = 0.0
    peak_used = 0
    req_id_counter = 0

    def admit():
        nonlocal req_id_counter
        while len(active) < max_concurrency and waiting:
            req_len = waiting.pop(0)
            n_blks  = math.ceil(req_len / block_size)
            try:
                alloc.allocate(req_id_counter, n_blks)
                active[req_id_counter] = req_len
                req_id_counter += 1
            except RuntimeError:
                # No blocks available — put request back and stop admitting
                waiting.insert(0, req_len)
                break

    admit()
    while active or waiting:
        if not active:
            admit()
        if not active:
            break

        t_ms += decode_ms_per_step
        finished = []
        for rid in list(active):
            active[rid] -= 1
            total_tokens += 1
            if active[rid] <= 0:
                finished.append(rid)

        for rid in finished:
            alloc.free(rid)
            del active[rid]
            done += 1

        peak_used = max(peak_used, alloc.blocks_used())
        admit()

    elapsed_s  = t_ms / 1000
    throughput = total_tokens / elapsed_s if elapsed_s > 0 else 0
    return {
        "throughput_tps": throughput,
        "total_ms":       t_ms,
        "real_tokens":    total_tokens,
        "peak_blocks_used": peak_used,
        "peak_utilisation": peak_used / total_blocks,
    }


random.seed(42)
sim_requests = [
    random.choices([50, 100, 200, 500],
                   weights=[0.35, 0.30, 0.25, 0.10])[0]
    for _ in range(128)
]

print(f"  {'Concurrency':>12}  {'Throughput':>12}  {'Peak blocks':>12}  {'Peak util%':>12}")
print(f"  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}")
for conc in [4, 8, 16, 32]:
    res = simulate_paged(sim_requests, conc, total_blocks, BLOCK_SIZE, 0.5)
    print(f"  {conc:>12}  {res['throughput_tps']:>10.0f}/s  "
          f"{res['peak_blocks_used']:>12}  {res['peak_utilisation']:>11.1%}")

# TODO 2: verify that concurrency=32 has higher throughput than concurrency=4
res4  = simulate_paged(sim_requests, 4,  total_blocks, BLOCK_SIZE, 0.5)
res32 = simulate_paged(sim_requests, 32, total_blocks, BLOCK_SIZE, 0.5)
assert res32["throughput_tps"] > res4["throughput_tps"], \
    "Higher concurrency should improve throughput"
print(f"\n  Concurrency 4 vs 32: {res4['throughput_tps']:.0f} vs {res32['throughput_tps']:.0f} tok/s")
print("  ✓ Section 4 passed — PagedAttention block allocation simulation working")


# ─────────────────────────────────────────────────────────────
# SECTION 5: Prefix caching — sharing blocks across requests
# ─────────────────────────────────────────────────────────────
print("\n── Section 5: Prefix Caching ──")
print("""
  Many requests share a common prefix (system prompt, few-shot examples).
  In standard PagedAttention, each request computes and stores its own
  copy of the prefix KV blocks — redundant work and memory.

  PREFIX CACHING (--enable-prefix-caching in vLLM):
    Blocks for an identical prefix are computed ONCE and shared across
    all requests that start with that prefix.

  Memory savings:
    N requests × L_prefix_tokens × KV_per_token
    → 1 copy shared = (N-1) × L_prefix_tokens × KV_per_token saved

  For a 512-token system prompt with Llama-7B FP16:
    Per request: 512 × 524,288 bytes = 256 MB
    10 concurrent requests: 10 × 256 MB = 2.56 GB
    With prefix caching: 1 × 256 MB = 2.3 GB saved

  Compute savings:
    Prefill for the shared prefix is skipped → lower TTFT for repeated prefixes.
""")

PREFIX_LEN   = 512
KV_PER_TOKEN_BYTES = 524_288
N_CONCURRENT = 10

without_cache = N_CONCURRENT * PREFIX_LEN * KV_PER_TOKEN_BYTES
with_cache    = 1            * PREFIX_LEN * KV_PER_TOKEN_BYTES
savings_gb    = (without_cache - with_cache) / 1e9

# TODO 3: Verify savings_gb > 0
assert savings_gb > 0, "Prefix caching should save memory"
print(f"  Prefix: {PREFIX_LEN} tokens, {N_CONCURRENT} concurrent requests (Llama-7B FP16)")
print(f"  Without prefix caching: {without_cache/1e9:.2f} GB")
print(f"  With prefix caching   : {with_cache/1e9:.2f} GB")
print(f"  Savings               : {savings_gb:.2f} GB  ({savings_gb/without_cache*100:.0f}%)")
print("  ✓ Section 5 passed — prefix caching shares KV blocks across requests")


print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 11.3 complete!")
print("  You implemented a block allocator, compared static vs paged")
print("  allocation, and measured the concurrency improvement.")
print("  Next: IV.LLM_Inference_Systems/12.Speculative_Decoding/12.1_draft_target_model.py")
print("=" * 60)
