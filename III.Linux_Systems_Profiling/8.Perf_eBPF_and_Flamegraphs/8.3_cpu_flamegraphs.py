#!/usr/bin/env python3
"""
8.Perf_eBPF_and_Flamegraphs/8.3_cpu_flamegraphs.py  ─  Chapter 8: CPU Flamegraphs
=======================================================================
Covers book section 8.2:
  • What a flamegraph shows — x-axis = time, y-axis = call depth
  • Identifying hotspots: wide frames at the top of a stack
  • Generating flamegraphs with perf+FlameGraph and py-spy
  • Differential flamegraphs for verifying optimizations
  • Classifying flamegraph frames by width percentage

Run:  python III.Linux_Systems_Profiling/8.Perf_eBPF_and_Flamegraphs/8.3_cpu_flamegraphs.py
All sections must print ✓.
"""

import time
import cProfile
import pstats
import io
import numpy as np

print("=" * 60)
print("  Exercise 8.2 — CPU Flamegraphs")
print("=" * 60)

DEVICE = "cpu"
print(f"  Device: {DEVICE}\n")

# ─────────────────────────────────────────────────────────────
# SECTION 1: Reading a Flamegraph
# ─────────────────────────────────────────────────────────────
print("── Section 1: Reading a Flamegraph ──")
print("""
  A flamegraph is a visualisation of sampled call stacks.
  The profiler interrupts the program thousands of times per second
  and records the full call stack at each sample.  After collecting
  all samples, they are sorted, merged, and drawn as stacked bars.

  HOW TO READ A FLAMEGRAPH:
    X-axis: alphabetical order within each level (NOT time order).
             Width of a frame = fraction of total samples where
             this function appeared in the call stack.
    Y-axis: call depth.  The bottom bar is the entry point.
             Each bar above is one level deeper in the call stack.

  FINDING HOTSPOTS:
    Wide frames at the TOP of their stack = this function is where
    the CPU actually spends time.  It owns CPU cycles.

    Tall thin stacks with no wide frames at the top = deep call
    chains where all the work happens further down.  The depth
    itself is not a problem — only wide tops are hotspots.

    A wide frame near the BOTTOM with nothing wide above it means
    the time is distributed among many callee functions — investigate
    each sub-frame individually.

  color is usually arbitrary (not meaningful) in standard flamegraphs.
  Some tools color-code by library (kernel = orange, user = yellow, JIT = green).

  Example: ASCII flamegraph of a PyTorch training pipeline:
  ─────────────────────────────────────────────────────────────
  [==========forward_pass (45%)=========][===backward (35%)===][dataload (20%)]
  [linear_1 ][attention][linear_2]        [grad_linear][grad_attn]
  [matmul   ][softmax  ]                  [matmul_grad]
  ─────────────────────────────────────────────────────────────

  Hotspot: forward_pass owns 45% of CPU samples.
  Within forward_pass: matmul is the deepest frame under linear_1
  — that is where cycles are actually spent.
  Action: profile with ncu to see whether this matmul is memory-bound
  or compute-bound (see Exercise 5.2).
""")

def find_hotspot(frame_widths: dict) -> str:
    """
    TODO 1: Implement this function.
    Given a dict mapping frame_name → width_pct (0–100),
    return the key with the maximum value.

    Example: find_hotspot({"forward": 45, "backward": 35, "dataload": 20})
    should return "forward" because 45 is the largest value.
    """
    pass  # YOUR CODE HERE → return max(frame_widths, key=frame_widths.get)


test_frames = {"forward": 45, "backward": 35, "dataload": 20}
result = find_hotspot(test_frames)
assert result == "forward", (
    f"find_hotspot({test_frames}) should return 'forward' (45%), "
    f"got '{result}'. Did you implement TODO 1?"
)

# Test edge cases
assert find_hotspot({"a": 10, "b": 90}) == "b", \
    "find_hotspot({'a':10, 'b':90}) should return 'b'"
assert find_hotspot({"only": 100}) == "only", \
    "find_hotspot with one key should return that key"

print(f"  find_hotspot({test_frames}) = '{result}'")
print("  ✓ Section 1 passed — hotspot identification function correct")


# ─────────────────────────────────────────────────────────────
# SECTION 2: Generating a Flamegraph with py-spy and cProfile
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: Generating a Flamegraph with py-spy and cProfile ──")
print("""
  py-spy is a Python profiler that samples the call stack without
  modifying your code.  It attaches to a running process or wraps
  a command and generates an SVG flamegraph.

  KEY ADVANTAGE over cProfile:
    py-spy uses OS-level sampling — it does NOT require you to add
    instrumentation to your code.  It works with any Python program,
    including ones that call into C extensions (numpy, PyTorch).
    It also works without sudo on most systems (no kernel privileges).

  py-spy commands:
    # Record a flamegraph while running a script:
    py-spy record -o flamegraph.svg -- python train.py

    # Attach to a running process by PID:
    py-spy record -o live_flame.svg --pid 12345

    # Top-like live view (no file, just terminal):
    py-spy top --pid 12345

  For this exercise, we use cProfile (built-in, always available)
  to profile a workload with a deliberate hotspot.  cProfile uses
  instrumentation rather than sampling — it counts every function
  call entry and exit.  It cannot profile C extension time (numpy
  and PyTorch show as single opaque calls), but it works everywhere.

  HOW TO RUN UNDER cProfile:
    python -m cProfile -s cumtime train.py | head -30

  The output table shows:
    ncalls  — how many times this function was called
    tottime — time in this function only (excluding callees)
    cumtime — total time including all functions this one called
    percall — time per call
""")

# Workload with a known slow function and a known fast function
def slow_matmul() -> float:
    """Heavy operation: 20 × 512×512 numpy matmul."""
    total = 0.0
    for _ in range(20):
        A = np.random.rand(512, 512).astype(np.float32)
        B = np.random.rand(512, 512).astype(np.float32)
        total += float(np.dot(A, B).sum())
    return total


def fast_scan() -> float:
    """Light operation: 20 × sequential sum on a small array."""
    total = 0.0
    for _ in range(20):
        arr = np.arange(10_000, dtype=np.float32)
        total += float(arr.sum())
    return total


def mixed_workload() -> None:
    """Calls both slow and fast functions — slow should dominate the profile."""
    _ = slow_matmul()
    _ = fast_scan()


# TODO 2: Profile mixed_workload() with cProfile and verify the profile ran.
#   Steps:
#     1. Create a cProfile.Profile() object
#     2. Call profile.enable(), run mixed_workload(), then profile.disable()
#     3. Create a pstats.Stats(profile) object and sort by 'cumulative'
#     4. Assert that the Stats object is not None
#   The Stats constructor validates the profile — if profiling failed it will raise.

profile = None  # YOUR CODE HERE → profile = cProfile.Profile()

assert profile is not None, (
    "profile must be a cProfile.Profile() object. Did you implement TODO 2?"
)

# Capture stats output (redirect stdout to a string buffer)
buf = io.StringIO()
stats = pstats.Stats(profile, stream=buf)
stats.sort_stats("cumulative")
stats.print_stats(5)
output = buf.getvalue()

assert "slow_matmul" in output or "fast_scan" in output or "mixed_workload" in output, (
    "cProfile output should contain function names from the workload. "
    f"Got: {output[:200]}"
)

print("  cProfile output (top 5 functions by cumulative time):")
for line in output.split("\n")[:15]:
    if line.strip():
        print(f"    {line}")
print("  ✓ Section 2 passed — cProfile ran successfully on mixed workload")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Differential Flamegraph
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: Differential Flamegraph ──")
print("""
  A differential flamegraph overlays two profiles captured before and
  after an optimization.  It shows which functions got faster (blue)
  and which got slower (red).

  This is critically important for verifying optimizations:
    1. You optimize a specific function, e.g., remove a sleep.
    2. You take a new profile.
    3. The differential flamegraph shows that only the target function
       changed (blue), while everything else stayed the same (grey).

  A common pitfall: optimizing function A shifts CPU time to function B,
  making B appear red.  Without a differential flamegraph, you might
  conclude B got slower.  The differential shows the shift is relative —
  B's absolute time did not change, but its fraction grew.

  TOOLS:
    # Using FlameGraph perl scripts:
    perf script -i slow.perf.data | stackcollapse-perf.pl > slow.folded
    perf script -i fast.perf.data | stackcollapse-perf.pl > fast.folded
    difffolded.pl slow.folded fast.folded | flamegraph.pl > diff.svg

    # Using py-spy differential:
    py-spy record -o slow.svg -- python slow_train.py
    py-spy record -o fast.svg -- python fast_train.py
    # (Open both SVGs — py-spy does not generate diffs natively)

  For this exercise, we profile a slow version (with sleep) and a fast
  version (no sleep) using cProfile, then compute the improvement.
""")

def profile_fn(fn) -> float:
    """
    TODO 3: Implement this function.
    Profile fn() using cProfile and return the total cumulative time
    in seconds for the top-level call.

    Steps:
      1. Create pr = cProfile.Profile()
      2. pr.enable(); fn(); pr.disable()
      3. Create stats = pstats.Stats(pr, stream=io.StringIO())
      4. stats.sort_stats("cumulative")
      5. Access stats.total_tt  (total time for all calls)
         OR: use stats.get_stats_profile().total_tt
         OR: sum the first 'tt' column — but the simplest is stats.total_tt
      Return the total time as a float.

    Hint: pstats.Stats has a .total_tt attribute after sort_stats is called.
    """
    pass  # YOUR CODE HERE → return total_time_seconds


slow_fn = lambda: time.sleep(0.02)
fast_fn = lambda: None

slow_time = profile_fn(slow_fn)
fast_time = profile_fn(fast_fn)

assert slow_time is not None, "profile_fn must return a float. Did you implement TODO 3?"
assert slow_time > 0.005, (
    f"profile_fn(lambda: time.sleep(0.02)) should return > 0.005 seconds, "
    f"got {slow_time}. Is cProfile measuring elapsed time?"
)
assert fast_time >= 0, "profile_fn(lambda: None) must return a non-negative time"

improvement = (slow_time - fast_time) / slow_time * 100 if slow_time > 0 else 0
print(f"  slow_fn profile time: {slow_time*1000:.1f} ms")
print(f"  fast_fn profile time: {fast_time*1000:.3f} ms")
print(f"  Improvement:          {improvement:.0f}%")
print("  ✓ Section 3 passed — differential profiling implemented")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Flamegraph Command Reference
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Flamegraph Command Reference ──")
print("""
  THREE TOOLS FOR GENERATING PYTHON FLAMEGRAPHS:

  ── Method 1: perf + FlameGraph (most detailed, requires sudo) ──
    # Install FlameGraph scripts:
    git clone https://github.com/brendangregg/FlameGraph /opt/FlameGraph

    # Record 30 seconds of call stack samples at 99 Hz:
    sudo perf record -F 99 -g -- python train.py --steps 200
    sudo perf record -F 99 -g -p $(pgrep python) sleep 30

    # Convert to flamegraph:
    sudo perf script | /opt/FlameGraph/stackcollapse-perf.pl > out.folded
    /opt/FlameGraph/flamegraph.pl out.folded > flamegraph.svg

    # Open in browser:
    xdg-open flamegraph.svg

    # For CPU-specific flamegraphs with correct Python symbols,
    # install: sudo apt install python3-dbg  (adds Python debug info)

  ── Method 2: py-spy (Python, no sudo) ──
    pip install py-spy

    # Record flamegraph of a full run:
    py-spy record -o flamegraph.svg -- python train.py

    # Record a sample at 100 Hz for 30 seconds:
    py-spy record --rate 100 --duration 30 -o flamegraph.svg -- python train.py

    # Attach to running process (no restart needed):
    py-spy record -o live.svg --pid $(pgrep -f train.py)

    # Top-like live view in terminal:
    py-spy top -- python train.py

  ── Method 3: torch.profiler Chrome trace (no sudo, PyTorch-integrated) ──
    from torch.profiler import profile, ProfilerActivity, schedule
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 on_trace_ready=torch.profiler.tensorboard_trace_handler('/tmp/tb'),
                 schedule=schedule(wait=1, warmup=1, active=3)) as prof:
        for step, (xb, yb) in enumerate(loader):
            # ... training step ...
            prof.step()

    # Open the Chrome trace at: chrome://tracing  →  Load  →  /tmp/tb/*.json
    # See Exercise 5.3 for full torch.profiler workflow.

  WHEN TO USE WHICH:
    perf flamegraph   → most accurate (hardware PMU), sees C extension internals
    py-spy flamegraph → easiest to use, Python-level visibility, no sudo
    torch.profiler    → PyTorch operator attribution, Chrome trace, Tensor Core info
""")

def classify_frame(width_pct: float) -> str:
    """
    TODO 4: Implement this function.
    Classify a flamegraph frame by its width percentage:
      width_pct > 20  → "hotspot"     (major consumer of CPU time)
      width_pct > 5   → "contributing" (notable but not dominant)
      else            → "noise"        (< 5% — too small to optimize)
    """
    pass  # YOUR CODE HERE → return classification string


assert classify_frame(35) == "hotspot",      f"35% → 'hotspot', got '{classify_frame(35)}'"
assert classify_frame(10) == "contributing", f"10% → 'contributing', got '{classify_frame(10)}'"
assert classify_frame(3)  == "noise",        f"3%  → 'noise', got '{classify_frame(3)}'"
assert classify_frame(21) == "hotspot",      f"21% → 'hotspot', got '{classify_frame(21)}'"
assert classify_frame(5)  == "noise",        f"5%  → exactly at boundary → 'noise', got '{classify_frame(5)}'"

print(f"  classify_frame(35) = '{classify_frame(35)}'")
print(f"  classify_frame(10) = '{classify_frame(10)}'")
print(f"  classify_frame(3)  = '{classify_frame(3)}'")
print("  ✓ Section 4 passed — frame classification function correct")


print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 8.2 complete!")
print()
print("  You can now read flamegraphs, identify hotspots, profile")
print("  Python code with cProfile, and compute improvement ratios")
print("  between slow and fast versions.")
print()
print("  Next: III.Linux_Systems_Profiling/8.Perf_eBPF_and_Flamegraphs/8.4_ebpf_and_bpftrace.py")
print("=" * 60)
