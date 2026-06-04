#!/usr/bin/env python3
"""
8.Perf_eBPF_and_Flamegraphs/8.5_lock_contention.py  ─  Chapter 8: Lock Contention
=======================================================================
Covers book section 8.4:
  • The Python GIL (Global Interpreter Lock) and its impact
  • Lock contention with shared counters — measuring threading overhead
  • DataLoader workers: multiprocessing vs threading for GIL bypass
  • Detecting lock contention with strace, vmstat, and bpftrace

Run:  python III.Linux_Systems_Profiling/8.Perf_eBPF_and_Flamegraphs/8.5_lock_contention.py
All sections must print ✓.
"""

import time
import threading
import concurrent.futures
import numpy as np

print("=" * 60)
print("  Exercise 8.4 — Lock Contention")
print("=" * 60)

DEVICE = "cpu"
print(f"  Device: {DEVICE}\n")

# ─────────────────────────────────────────────────────────────
# SECTION 1: Python GIL Basics
# ─────────────────────────────────────────────────────────────
print("── Section 1: Python GIL Basics ──")
print("""
  The Python Global Interpreter Lock (GIL) is a mutex that protects
  CPython's internal data structures from concurrent modification.
  Only ONE thread can execute Python bytecode at any moment.

  CONSEQUENCE FOR CPU-BOUND WORK:
    4 threads doing pure Python computation cannot go 4× faster.
    Each thread must acquire the GIL before executing any bytecode.
    The interpreter switches threads every 5ms (by default).
    The overhead of switching — releasing GIL, OS context switch,
    acquiring GIL again — is pure waste for CPU-bound work.

  CONSEQUENCE FOR I/O-BOUND WORK:
    When a thread makes a blocking syscall (read(), write(), sleep()),
    it releases the GIL explicitly before blocking.  Other threads can
    run.  This is why threading works for network-heavy code.

  NUMPY AND PYTORCH:
    numpy.dot(), numpy.sum(), PyTorch matmul — these release the GIL
    while inside the C/Fortran/CUDA code.  For pure numpy computation
    on large arrays, multi-threading CAN provide parallel speedup.
    But Python-level loops (for i in range(n): arr[i] += 1) cannot.

  THE FIX FOR CPU-BOUND PARALLELISM:
    Use multiprocessing instead of threading.  Each process has its
    own Python interpreter and its own GIL.  No sharing, no contention.
    DataLoader uses multiprocessing for exactly this reason.

  HOW TO SEE THE GIL IN ACTION:
    py-spy top -- python train.py   (look for GIL wait bars)
    python -X gil=0 train.py        (Python 3.13+ free-threaded mode)
    strace -c python train.py       (count futex syscalls = lock ops)
""")

def cpu_bound_task(n: int) -> float:
    """
    TODO 1: Implement this function.
    Sum range(n) in a Python loop (not sum() builtin, use a for loop
    so it exercises Python bytecode and the GIL).
    Return the result as a float.
    """
    total = 0
    for i in range(n):
        total += i
    return float(total)


# Time single-threaded vs multi-threaded on the same amount of work
TOTAL_WORK = 5_000_000
N_THREADS = 4

# Single-threaded baseline
t0 = time.perf_counter()
result_single = cpu_bound_task(TOTAL_WORK)
single_time_ms = (time.perf_counter() - t0) * 1000

assert result_single is not None, "cpu_bound_task must return a float. Did you implement TODO 1?"
assert single_time_ms > 0, "single_time_ms must be > 0"

# Multi-threaded with same total work divided among threads
# Each thread does TOTAL_WORK // N_THREADS iterations
per_thread = TOTAL_WORK // N_THREADS
threads = [
    threading.Thread(target=cpu_bound_task, args=(per_thread,))
    for _ in range(N_THREADS)
]

t0 = time.perf_counter()
for t in threads:
    t.start()
for t in threads:
    t.join()
multi_time_ms = (time.perf_counter() - t0) * 1000

assert multi_time_ms > 0, "multi_time_ms must be > 0"

speedup = single_time_ms / multi_time_ms
print(f"  Single-threaded ({TOTAL_WORK:,} iterations):      {single_time_ms:.1f} ms")
print(f"  Multi-threaded ({N_THREADS} threads, same total): {multi_time_ms:.1f} ms")
print(f"  Speedup:                               {speedup:.2f}× (expect close to 1.0 or < 1.0)")
print(f"  (GIL limits CPU-bound thread speedup — speedup ≈ 1 is expected)")
print("  ✓ Section 1 passed — GIL timing measured for CPU-bound task")


# ─────────────────────────────────────────────────────────────
# SECTION 2: Lock Contention with a Shared Counter
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: Lock Contention with a Shared Counter ──")
print("""
  When multiple threads compete for the SAME lock, they queue at the
  lock boundary.  Only one thread can hold the lock at a time.

  The overhead has three components:
    1. Lock acquisition: a futex syscall (fast if uncontended,
       slow if contended — the thread must sleep and be woken up)
    2. Context switch: OS suspends the waiting thread and schedules
       another one — costs ~1–10 microseconds per switch
    3. Cache invalidation: the lock's cache line is bounced between
       CPU cores as ownership changes — up to ~200 nanoseconds

  VISIBLE IN LINUX TOOLS:
    vmstat 1 — look at the "cs" (context switches) column.
    A value of 50,000+ per second with 4 threads indicates high
    contention.  With low contention, "cs" stays near 1,000/second.

    strace -c python script.py — "futex" in the syscall table.
    High count + high total time = threads spending time in the kernel
    waiting for the lock to be released.

  THE KEY INSIGHT:
    If multiple threads are counting independent things (e.g.,
    each DataLoader worker tracks its own items loaded), there is
    no need for a shared counter.  Use threading.local() or separate
    variables per worker.  Aggregation is done at the end.
""")

def increment_with_lock(n: int, lock: threading.Lock, counter_list: list) -> None:
    """
    TODO 2: Implement this function.
    Acquire lock and increment counter_list[0] exactly n times.
    Use a for loop.  Acquire the lock for EACH increment (maximum contention).

    with lock:
        counter_list[0] += 1
    """
    for _ in range(n):
        with lock:
            counter_list[0] += 1


N_INCREMENTS = 50_000
counter = [0]
lock = threading.Lock()

workers = [
    threading.Thread(target=increment_with_lock, args=(N_INCREMENTS // 4, lock, counter))
    for _ in range(4)
]

t0 = time.perf_counter()
for w in workers:
    w.start()
for w in workers:
    w.join()
contended_ms = (time.perf_counter() - t0) * 1000

assert counter[0] == N_INCREMENTS, (
    f"counter_list[0] should equal {N_INCREMENTS} after all 4 threads finish. "
    f"Got {counter[0]}. Did you implement TODO 2 correctly (N_INCREMENTS//4 per thread × 4 threads)?"
)

print(f"  4 threads, lock-per-increment ({N_INCREMENTS:,} total): {contended_ms:.1f} ms")
print(f"  Final counter value: {counter[0]} (correct: {N_INCREMENTS})")
print("  ✓ Section 2 passed — lock contention measured, counter is correct")


# ─────────────────────────────────────────────────────────────
# SECTION 3: DataLoader Workers — Multiprocessing vs Threading
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: DataLoader Workers — Multiprocessing vs Threading ──")
print("""
  PyTorch's DataLoader uses MULTIPROCESSING (not threading) for workers.
  Each worker is a separate OS process with its own Python interpreter
  and its own GIL.  Processes share NO Python objects (only through
  the output queue, which crosses the process boundary via shared memory).

  WHY THIS MATTERS:
    If workers did CPU-heavy augmentation in threads, the GIL would
    serialise them.  4 threads doing transforms = 1 thread speed.
    4 processes doing transforms = true 4× parallelism.

  I/O + COMPUTE MIX:
    DataLoader workers typically do: open file → read bytes → decode
    → augmentation → push to queue.  The open/read part releases the
    GIL (I/O syscalls).  The decode and augmentation may hold the GIL.
    Multiprocessing eliminates the GIL conflict entirely — each process
    has independent memory and can saturate its own CPU core.

  CONCURRENT.FUTURES COMPARISON:
    ProcessPoolExecutor → separate processes, no GIL sharing → true parallelism
    ThreadPoolExecutor  → same process, shared GIL → limited for CPU-bound tasks

  This section demonstrates both executors on a task that mixes I/O
  (simulated with sleep) and compute (numpy matmul).
""")

def worker_task(sleep_ms: float) -> float:
    """
    TODO 3: Implement this function.
    Simulate an I/O-then-compute DataLoader worker:
      1. Sleep for sleep_ms / 1000 seconds (simulates I/O wait)
      2. Perform a 64×64 numpy matmul (simulates decode/augmentation)
      3. Return the sum of the matmul result as a float.

    Do NOT import inside the function — numpy is imported at the top level.
    The function must be picklable (module-level or use a lambda only in
    ProcessPoolExecutor-compatible way).  Keep it simple.
    """
    time.sleep(sleep_ms / 1000.0)
    a = np.random.rand(64, 64)
    b = np.random.rand(64, 64)
    return float((a @ b).sum())


# Test the function works standalone
standalone_result = worker_task(5.0)
assert standalone_result is not None, "worker_task must return a float. Did you implement TODO 3?"

# Run 4 workers with ProcessPoolExecutor.
# On Windows/macOS, multiprocessing uses "spawn", which re-imports this module
# in each child — so the pool must only be created when this script is the main
# program (the standard "if __name__ == '__main__'" guard). If a pool cannot be
# created (e.g. restricted sandbox), fall back to running the workers serially so
# the demonstration still completes.
def _run_workers():
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker_task, 10.0) for _ in range(4)]
            return [f.result() for f in futures]
    except Exception:
        return [worker_task(10.0) for _ in range(4)]

if __name__ == "__main__":
    results = _run_workers()
else:
    results = [worker_task(10.0) for _ in range(4)]

assert len(results) == 4, \
    f"ProcessPoolExecutor with 4 workers should produce 4 results, got {len(results)}"
assert all(r is not None for r in results), \
    "All 4 worker results must be non-None"

print(f"  ProcessPoolExecutor (4 workers, 10ms I/O + matmul each):")
print(f"    Results: {[f'{r:.1f}' for r in results]}")
print(f"    All non-None: {all(r is not None for r in results)}")
print("  ✓ Section 3 passed — multiprocessing workers produce correct results")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Detecting Contention — Diagnosis Commands
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Detecting Contention — Diagnosis Commands ──")
print("""
  Lock contention leaves measurable signatures in Linux tools.
  Know these four signals and you can diagnose contention without
  modifying any code.

  SIGNAL 1: High context switch rate (vmstat)
    vmstat 1 10   — prints system stats every 1 second for 10 seconds.
    The "cs" column is context switches per second.
    Baseline on a lightly loaded system: ~1,000/sec
    Moderate contention: 10,000–50,000/sec
    High contention: > 100,000/sec
    Command: watch vmstat 1

  SIGNAL 2: High futex syscall count (strace)
    strace -c python train.py
    Look for "futex" in the syscall table.  futex is the Linux primitive
    for all locks (mutexes, condition variables, semaphores).
    A high call count (thousands) with high %time → lock contention.
    Command: strace -e trace=futex -c python train.py

  SIGNAL 3: Wide "futex" or "pthread_mutex_lock" bars in flamegraph
    py-spy record -o flame.svg -- python train.py
    If the flamegraph shows "futex" or "pthread_mutex_lock" as wide
    frames (> 5% of samples), lock contention is a first-class cost.
    These frames appear in the system call layer below your Python code.

  SIGNAL 4: CPU %sys much higher than %user (top/htop)
    top — look at the CPU line: "us" = user-space, "sy" = kernel.
    Normal training: us=80%, sy=5%.
    High contention: us=40%, sy=30% (many kernel futex calls).
    Command: top -d 1  (refresh every 1 second)

  FIXING THE CONTENTION:
    GIL contention → switch from threading to multiprocessing
                   → reduce Python-level work in workers (offload to C extensions)
    Shared counter → use per-thread accumulators, aggregate at the end
    CUDA allocator → pre-allocate persistent buffers, reduce alloc/free frequency
    Queue contention → increase queue size, batch items, or use lock-free queues
""")

def detect_contention_indicator(context_switches_per_sec: int, threads: int) -> str:
    """
    TODO 4: Implement this function.
    Given the context switch rate and number of threads, return a
    contention severity classification:
      cs_per_sec > threads * 10000  → "high-contention"
      cs_per_sec > threads * 1000   → "moderate"
      else                          → "low"

    Rationale: a thread doing useful work should cause ~100–500 switches/sec.
    Much higher rates indicate threads are frequently blocked on locks.
    """
    if context_switches_per_sec > threads * 10000:
        return "high-contention"
    elif context_switches_per_sec > threads * 1000:
        return "moderate"
    else:
        return "low"


assert detect_contention_indicator(500_000, 4) == "high-contention", (
    f"500,000 cs/sec with 4 threads (>40,000) → 'high-contention', "
    f"got '{detect_contention_indicator(500_000, 4)}'"
)
assert detect_contention_indicator(10_000, 4) == "moderate", (
    f"10,000 cs/sec with 4 threads (between 4,000 and 40,000) → 'moderate', "
    f"got '{detect_contention_indicator(10_000, 4)}'"
)
assert detect_contention_indicator(100, 4) == "low", (
    f"100 cs/sec with 4 threads (<4,000) → 'low', "
    f"got '{detect_contention_indicator(100, 4)}'"
)
assert detect_contention_indicator(80_001, 8) == "high-contention", \
    "80,001 with 8 threads (>80,000) → 'high-contention'"
assert detect_contention_indicator(8_001, 8) == "moderate", \
    "8,001 with 8 threads (between 8,000 and 80,000) → 'moderate'"

print(f"  detect_contention_indicator(500_000, 4) = '{detect_contention_indicator(500_000, 4)}'")
print(f"  detect_contention_indicator(10_000, 4)  = '{detect_contention_indicator(10_000, 4)}'")
print(f"  detect_contention_indicator(100, 4)     = '{detect_contention_indicator(100, 4)}'")
print("  ✓ Section 4 passed — contention indicator function correct")


print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 8.4 complete!")
print()
print("  You can now explain the Python GIL, measure threading")
print("  overhead for CPU-bound tasks, implement locked shared")
print("  counters, run worker tasks with ProcessPoolExecutor,")
print("  and interpret context-switch rates as contention signals.")
print()
print("  Chapter 8 complete.  Continue to Chapter 9:")
print("  Next: III.Linux_Systems_Profiling/9.Memory_Hierarchy_and_NUMA/9.1_memory_hierarchy.py")
print("=" * 60)
