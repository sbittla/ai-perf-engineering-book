#!/usr/bin/env python3
"""
Appendices/E.Advanced_Capstones/E.2_vllm_trace.py  --  Appendix E.2
=======================================================================
vLLM Internals: simulate the continuous-batching scheduler and PagedAttention
block table so you know what to look for when you trace the real engine.

Difficulty: ****-  (4/5 - simulation; real trace is the live extension)
Est. time:  8-12 hours for the instrumented vLLM trace
Expected ranges:
  - Continuous batching keeps GPU slot occupancy far higher than static batching
  - Continuous throughput >> static throughput on mixed-length workloads
Troubleshooting:
  - No GPU or vLLM needed here; this models the scheduler decisions you trace.
Live run (the real capstone):
  - Instrument vLLM and follow ONE request through admission -> prefill ->
    scheduler -> PagedAttention block alloc -> decode. Produce an nsys timeline.

Run:  python Appendices/E.Advanced_Capstones/E.2_vllm_trace.py
"""
print("=" * 70)
print("  Exercise E.2 - vLLM Internals (scheduler + paged KV simulation)")
print("=" * 70)

# A mixed workload: requests arrive with different decode lengths.
# (request_id -> tokens to generate)
REQS = {0: 8, 1: 40, 2: 12, 3: 60, 4: 20, 5: 6, 6: 30, 7: 50}
SLOTS = 4   # max concurrent sequences on the GPU

# ---------------------------------------------------------------------------
# SECTION 1: Static batching - the whole batch waits for the slowest member
# ---------------------------------------------------------------------------
print("\n-- Section 1: Static Batching --")
print("""
  Static batching runs a fixed batch to completion: every slot is occupied
  until the LONGEST sequence in the batch finishes, wasting the freed slots.
""")

reqs = list(REQS.items())
static_steps = 0
static_slot_token_capacity = 0
for i in range(0, len(reqs), SLOTS):
    batch = reqs[i:i + SLOTS]
    longest = max(t for _, t in batch)
    static_steps += longest
    static_slot_token_capacity += longest * SLOTS   # slots held this long
static_useful = sum(REQS.values())
static_util = static_useful / static_slot_token_capacity

print(f"  Requests        : {len(REQS)}  | slots: {SLOTS}")
print(f"  Total steps     : {static_steps}")
print(f"  Slot utilization: {static_util:6.1%}  (useful tokens / slot-steps held)")

# ---------------------------------------------------------------------------
# SECTION 2: Continuous batching - free slots refill immediately
# ---------------------------------------------------------------------------
print("\n-- Section 2: Continuous Batching --")
print("""
  Continuous batching evicts a finished sequence the moment it emits its last
  token and admits a waiting request into that slot the next step.
""")

remaining = dict(REQS)
waiting = list(REQS.keys())
active = {}          # slot -> request_id
cont_steps = 0
slot_steps_used = 0
while waiting or active:
    # fill free slots
    while len(active) < SLOTS and waiting:
        rid = waiting.pop(0)
        free = next(s for s in range(SLOTS) if s not in active.values())
        active[free] = rid
    # one decode step for every active slot
    cont_steps += 1
    slot_steps_used += len(active)
    done = []
    for slot, rid in active.items():
        remaining[rid] -= 1
        if remaining[rid] <= 0:
            done.append(slot)
    for slot in done:
        del active[slot]

cont_util = sum(REQS.values()) / (cont_steps * SLOTS)
print(f"  Total steps     : {cont_steps}")
print(f"  Slot utilization: {cont_util:6.1%}")
speedup = static_steps / cont_steps
print(f"  Step speedup vs static: {speedup:.2f}x")
assert cont_steps < static_steps and cont_util > static_util
print("  [check] continuous batching finishes sooner with higher slot utilization")

# ---------------------------------------------------------------------------
# SECTION 3: PagedAttention block table for one request
# ---------------------------------------------------------------------------
print("\n-- Section 3: PagedAttention Block Allocation --")
print("""
  Each sequence's KV cache is stored in fixed-size blocks, allocated lazily as
  the sequence grows - no up-front reservation of max_seq_len.
""")

BLOCK = 16          # tokens per KV block
prompt_len = 40
gen_len = 60
import math
blocks_after_prefill = math.ceil(prompt_len / BLOCK)
blocks_final = math.ceil((prompt_len + gen_len) / BLOCK)
naive_blocks = math.ceil(2048 / BLOCK)   # if we pre-reserved max_seq_len=2048
print(f"  Block size            : {BLOCK} tokens")
print(f"  Blocks after prefill  : {blocks_after_prefill}")
print(f"  Blocks at completion  : {blocks_final}")
print(f"  Naive pre-reservation : {naive_blocks} blocks (max_seq_len=2048)")
saved = 1 - blocks_final / naive_blocks
print(f"  Memory saved vs naive : {saved:6.1%}")
assert blocks_final < naive_blocks
print("  [check] lazy paged allocation avoids reserving the worst-case context")

print("\n" + "=" * 70)
print("  ALL SECTIONS PASSED - Exercise E.2 complete")
print("  You simulated the scheduler and the block table. Now trace the real")
print("  vLLM engine and confirm these decisions in an nsys timeline.")
print("=" * 70)
