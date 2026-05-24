#!/usr/bin/env bash
# =============================================================================
# profile_nsys.sh  —  Project 1, Step 2: Nsight Systems GPU Timeline Profiling
# =============================================================================
# PURPOSE:
#   Wrap baseline_inference.py with Nsight Systems (nsys) to capture a
#   full system-wide performance timeline: CPU activity, CUDA kernel launches,
#   memory transfers (H2D/D2H), and GPU utilisation — all on one timeline.
#
# WHAT nsys SHOWS YOU:
#   - How long each CUDA kernel runs on the GPU
#   - Host-to-Device (H2D) and Device-to-Host (D2H) memory transfer durations
#   - CPU↔GPU synchronisation points (where the CPU waits for GPU)
#   - Gaps between GPU operations (GPU sitting idle = CPU-bound bottleneck)
#   - NVTX markers if you added them in your Python code
#
# HOW TO READ THE REPORT:
#   Open the .nsys-rep file in the Nsight Systems GUI:
#     nsys-ui reports/baseline.nsys-rep
#   Or read the text summary printed at the end of this script.
#
# KEY THINGS TO LOOK FOR:
#   1. Long gaps between CUDA kernels → CPU is the bottleneck (DataLoader, tokenizer)
#   2. H2D transfers taking a long time → data not in pinned memory
#   3. Low SM Activity % → GPU underutilised (small batch or memory-bound)
#   4. Many tiny kernels instead of few large ones → kernel launch overhead
#
# REQUIREMENTS:
#   nsys installed: /usr/local/cuda/bin/nsys or from Nsight Systems installer
#   Verify: nsys --version
# =============================================================================

set -euo pipefail

MODEL="${1:-gpt2}"          # First argument = model name, default gpt2
OUTDIR="reports"
mkdir -p "$OUTDIR"

echo "============================================================"
echo "  Nsight Systems Profiling — $MODEL"
echo "============================================================"

# ── Profile 1: Quick stats summary (no GUI needed) ───────────────────────────
echo ""
echo "[1/3] Running with --stats=true (prints summary to terminal)..."
echo "      This shows top CUDA kernels by time, memory transfers, API calls."
echo ""

# --stats=true         : print a text summary after the run
# --trace=cuda         : capture CUDA API calls and kernel launches
# --trace=osrt         : capture OS runtime calls (threading, file I/O)
# --cuda-memory-usage  : track GPU memory allocations over time
# python ...           : the application being profiled (run with small token count for speed)
nsys profile \
    --stats=true \
    --trace=cuda,osrt \
    --cuda-memory-usage=true \
    --output="$OUTDIR/baseline_stats" \
    python baseline_inference.py --model "$MODEL" --tokens 50 --runs 1

echo ""
echo "[1/3] Done. Stats printed above."

# ── Profile 2: Full timeline capture for GUI ─────────────────────────────────
echo ""
echo "[2/3] Capturing full timeline report (for Nsight Systems GUI)..."
echo "      Output: $OUTDIR/baseline_full.nsys-rep"
echo ""

# -o <name>            : output filename (without extension)
# --trace=cuda,nvtx,osrt,cublas,cudnn : include all relevant subsystems
#   cuda    — kernel launches and memory ops
#   nvtx    — named ranges you add with torch.cuda.nvtx.range_push/pop()
#   osrt    — OS threads and synchronisation
#   cublas  — cuBLAS calls (matrix multiplies)
#   cudnn   — cuDNN calls (convolutions, attention in older PyTorch)
nsys profile \
    --trace=cuda,nvtx,osrt,cublas,cudnn \
    --cuda-memory-usage=true \
    --output="$OUTDIR/baseline_full" \
    python baseline_inference.py --model "$MODEL" --tokens 100 --runs 2

echo ""
echo "[2/3] Done. Open in GUI: nsys-ui $OUTDIR/baseline_full.nsys-rep"

# ── Profile 3: Snapshot a "live" inference server ────────────────────────────
# This simulates profiling a production server that's already running.
# You start it, wait for it to warm up, then attach.
echo ""
echo "[3/3] Demonstrating delayed snapshot (simulates live server profiling)..."
echo "      Starts the script, waits 2s (warmup), profiles for 5s."
echo ""

# --delay=N            : wait N seconds after process starts before recording
# --duration=N         : record for N seconds then stop (don't wait for process end)
# Useful for: profiling a server that runs indefinitely; you snapshot the steady state
nsys profile \
    --delay=2 \
    --duration=5 \
    --trace=cuda,osrt \
    --output="$OUTDIR/baseline_snapshot" \
    python baseline_inference.py --model "$MODEL" --tokens 200 --runs 10 || true
# '|| true' because nsys will kill the process after 5s which returns non-zero

echo ""
echo "[3/3] Done. Snapshot: $OUTDIR/baseline_snapshot.nsys-rep"

# ── Print CLI analysis of the full report ────────────────────────────────────
echo ""
echo "============================================================"
echo "  CLI Analysis of baseline_full.nsys-rep"
echo "============================================================"
echo ""
echo "── GPU Kernel Summary ──"
# --report gputrace   : shows each kernel: name, duration, SM occupancy
nsys analyze "$OUTDIR/baseline_full.nsys-rep" --report gputrace 2>/dev/null || \
    echo "  (nsys analyze not available — use nsys-ui to view)"

echo ""
echo "── Files generated ──"
ls -lh "$OUTDIR"/*.nsys-rep 2>/dev/null || echo "  (no .nsys-rep files found)"

echo ""
echo "============================================================"
echo "  HOW TO INTERPRET (look for these in the GUI timeline):"
echo ""
echo "  ✓ Healthy: GPU row shows solid green, few gaps"
echo "  ✗ Problem: Large white gaps in GPU row = CPU-bound bottleneck"
echo "  ✗ Problem: Many tiny kernels = kernel launch overhead"
echo "  ✗ Problem: Long orange H2D bars = slow data transfers"
echo ""
echo "  Next step: profile_ncu.sh to drill into individual kernels"
echo "============================================================"
