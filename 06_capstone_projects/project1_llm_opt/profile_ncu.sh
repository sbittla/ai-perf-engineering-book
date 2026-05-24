#!/usr/bin/env bash
# =============================================================================
# profile_ncu.sh  —  Project 1, Step 3: Nsight Compute Per-Kernel Analysis
# =============================================================================
# PURPOSE:
#   After nsys shows WHICH kernels are slow, ncu explains WHY.
#   Nsight Compute collects hundreds of hardware counters per-kernel:
#     - Memory bandwidth utilisation (how close to peak?)
#     - SM (Streaming Multiprocessor) occupancy (warps active vs max)
#     - Warp efficiency (what fraction of threads do useful work?)
#     - Instruction throughput
#     - Roofline position: are you memory-bound or compute-bound?
#
# MEMORY-BOUND vs COMPUTE-BOUND (key concept):
#   Every kernel either:
#   (a) Spends most time waiting for memory (bandwidth limited) — memory-bound
#   (b) Spends most time doing arithmetic (FLOPs limited) — compute-bound
#   
#   For LLMs:
#     Prefill (processing input tokens) → usually compute-bound (large matmuls)
#     Decode (generating tokens one-by-one) → usually memory-bound (KV cache reads)
#
# WARNING: ncu is SLOW (10–100x slowdown) because it injects counters.
#          Use it only on the specific kernel you want to analyse.
#
# REQUIREMENTS:
#   ncu installed: /usr/local/cuda/bin/ncu
#   Verify: ncu --version
# =============================================================================

set -euo pipefail

MODEL="${1:-gpt2}"
OUTDIR="reports"
mkdir -p "$OUTDIR"

echo "============================================================"
echo "  Nsight Compute Per-Kernel Analysis — $MODEL"
echo "============================================================"

# ── Step 1: List what metrics are available on this GPU ──────────────────────
echo ""
echo "[0] Available memory metrics on this GPU:"
echo "    (useful for building custom metric sets)"
echo ""
# --query-metrics lists every counter the GPU exposes
# grep filters to memory-related ones
ncu --query-metrics 2>/dev/null | grep -i "memory" | head -20 || \
    echo "  (ncu not found — install Nsight Compute)"

# ── Step 2: Quick summary — all kernels, minimal metrics ─────────────────────
echo ""
echo "[1/4] Profiling all kernels with default metrics..."
echo "      Expect: 5–20x slower than normal execution"
echo ""

# --set basic         : collect a small, fast metric set
# --csv               : output in CSV format for programmatic parsing
# --log-file          : save output to file as well as terminal
ncu \
    --set basic \
    --csv \
    --log-file "$OUTDIR/ncu_basic.csv" \
    python baseline_inference.py --model "$MODEL" --tokens 20 --runs 1 \
    2>/dev/null | head -30 || true

echo "  Saved: $OUTDIR/ncu_basic.csv"

# ── Step 3: Full metrics on attention kernels only ───────────────────────────
echo ""
echo "[2/4] Deep-profiling attention/matmul kernels (--set full)..."
echo "      Filters to kernels with 'mm' or 'attention' in name"
echo ""

# --set full          : collect ALL available metrics (very slow but complete)
# --kernel-name regex : only profile kernels whose name matches this regex
# --launch-count 3    : only profile the first 3 invocations of matching kernels
# This prevents profiling hundreds of identical kernel calls
ncu \
    --set full \
    --kernel-name ".*mm.*|.*attention.*|.*gemm.*" \
    --launch-count 3 \
    --output "$OUTDIR/ncu_attention" \
    python baseline_inference.py --model "$MODEL" --tokens 20 --runs 1 \
    2>/dev/null || echo "  (kernel name may not match — check with --set basic first)"

echo "  Saved: $OUTDIR/ncu_attention.ncu-rep (open with ncu-ui)"

# ── Step 4: Specific roofline metrics ────────────────────────────────────────
echo ""
echo "[3/4] Collecting roofline metrics (bandwidth + compute utilisation)..."
echo "      These tell you exactly: memory-bound or compute-bound?"
echo ""

# These specific metrics form the basis of a Roofline model:
#
# sm__throughput: what % of peak SM throughput are you achieving?
# dram__throughput: what % of peak memory bandwidth are you using?
# l1tex__throughput: L1/shared memory throughput utilisation
# sm__warps_active: avg active warps per SM per cycle (occupancy proxy)
#
# If dram__throughput is ~100% and sm__throughput is low → MEMORY-BOUND
# If sm__throughput is ~100% and dram__throughput is low → COMPUTE-BOUND
ROOFLINE_METRICS="\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active"

ncu \
    --metrics "$ROOFLINE_METRICS" \
    --kernel-name ".*mm.*" \
    --launch-count 2 \
    --csv \
    --log-file "$OUTDIR/ncu_roofline.csv" \
    python baseline_inference.py --model "$MODEL" --tokens 10 --runs 1 \
    2>/dev/null || true

echo "  Saved: $OUTDIR/ncu_roofline.csv"
echo ""
echo "  HOW TO READ ROOFLINE RESULTS:"
echo "  dram__throughput ≈ 100% → memory-bound  (need to reduce memory access)"
echo "  sm__throughput   ≈ 100% → compute-bound (need more FLOPs throughput)"

# ── Step 5: Save report for GUI ──────────────────────────────────────────────
echo ""
echo "[4/4] Saving full report for Nsight Compute GUI..."

# --import-source yes : includes source code correlation in the report
# Open with: ncu-ui reports/ncu_full.ncu-rep
ncu \
    --set roofline \
    --import-source yes \
    --launch-count 2 \
    --output "$OUTDIR/ncu_full" \
    python baseline_inference.py --model "$MODEL" --tokens 10 --runs 1 \
    2>/dev/null || true

echo ""
echo "============================================================"
echo "  Files generated:"
ls -lh "$OUTDIR"/ncu* 2>/dev/null || echo "  (no files — ncu may not be installed)"
echo ""
echo "  Open GUI: ncu-ui $OUTDIR/ncu_full.ncu-rep"
echo ""
echo "  KEY METRICS TO CHECK:"
echo "  1. sm__warps_active: <50% = low occupancy (batch too small?)"
echo "  2. dram throughput vs sm throughput → identifies bound type"
echo "  3. L1 cache hit rate → low = memory access pattern is poor"
echo "============================================================"
