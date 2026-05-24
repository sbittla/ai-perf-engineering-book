#!/usr/bin/env bash
# =============================================================================
# profile_flamegraph.sh  —  Project 4: Generate Flamegraph SVGs
# =============================================================================
# PURPOSE:
#   Profile slow_training.py or fast_training.py using py-spy to generate
#   a flamegraph SVG. The flamegraph visually shows where CPU time is spent.
#
# HOW TO READ A FLAMEGRAPH:
#   - X axis = time (wider = more CPU time)
#   - Y axis = call stack depth (bottom = bottom of stack, top = leaf function)
#   - Each rectangle = one function in the call stack
#   - Width = fraction of total time spent in that function + its callees
#   - LOOK FOR: wide rectangles at the TOP of a stack = hot leaf functions
#   - LOOK FOR: wide rectangles with narrow children = function itself is slow
#
# DIFFERENTIAL FLAMEGRAPH (run diff_flamegraph.sh after):
#   - Red   = function got SLOWER (appeared MORE after the change)
#   - Blue  = function got FASTER (appeared LESS after the change)
#   - Width = magnitude of change
#   A good optimisation shows blue where the bottleneck was, red nowhere.
#
# REQUIREMENTS:
#   py-spy:      pip install py-spy
#   FlameGraph:  git clone https://github.com/brendangregg/FlameGraph ~/FlameGraph
#
# HOW TO USE:
#   bash profile_flamegraph.sh slow    # profiles slow_training.py
#   bash profile_flamegraph.sh fast    # profiles fast_training.py
# =============================================================================

set -euo pipefail

TARGET="${1:-slow}"   # 'slow' or 'fast'
FLAMEGRAPH_DIR="${FLAMEGRAPH_DIR:-$HOME/FlameGraph}"
OUTDIR="reports"
mkdir -p "$OUTDIR"

# Select which script to profile
case "$TARGET" in
    slow) SCRIPT="slow_training.py --steps 80" ;;
    fast) SCRIPT="fast_training.py --steps 80" ;;
    *)    echo "Usage: $0 slow|fast"; exit 1 ;;
esac

echo "============================================================"
echo "  Flamegraph Profiling: $TARGET"
echo "============================================================"

# =============================================================================
# METHOD 1: py-spy (recommended — Python-aware, no root needed in most cases)
# =============================================================================
if command -v py-spy &>/dev/null; then
    echo ""
    echo "[1/2] py-spy flamegraph..."
    echo ""
    echo "  WHAT py-spy DOES:"
    echo "  - Samples Python stack frames at 100Hz (every 10ms)"
    echo "  - No code instrumentation required — attaches to any running Python"
    echo "  - Python-aware: shows function names, not just memory addresses"
    echo "  - Includes both Python frames AND C extension frames (PyTorch ops)"
    echo ""
    
    SVG_PATH="$OUTDIR/${TARGET}_pyspy.svg"
    FOLDED_PATH="$OUTDIR/${TARGET}_pyspy.folded"
    
    # py-spy record:
    #   -o <file>      : output file (.svg for flamegraph, .txt for folded stacks)
    #   --rate <hz>    : samples per second (100 = 10ms resolution)
    #   --subprocesses : also profile child processes (DataLoader workers)
    #   --threads      : show all threads, not just main thread
    #   -- <cmd>       : the command to run and profile
    py-spy record \
        --output "$SVG_PATH" \
        --format speedscope \
        --rate 100 \
        --subprocesses \
        --threads \
        -- python $SCRIPT
    
    echo "  ✓ SVG flamegraph: $SVG_PATH"
    echo "     Open in browser: firefox $SVG_PATH"
    
    # Also save folded format for differential flamegraph
    py-spy record \
        --output "$FOLDED_PATH" \
        --format raw \
        --rate 100 \
        -- python $SCRIPT
    
    echo "  ✓ Folded stacks:  $FOLDED_PATH (used for diff flamegraph)"

else
    echo "  py-spy not found. Install: pip install py-spy"
fi

# =============================================================================
# METHOD 2: perf + FlameGraph (system-level, includes kernel frames)
# =============================================================================
echo ""
echo "[2/2] perf + FlameGraph (requires perf + FlameGraph scripts)..."
echo ""
echo "  WHAT perf ADDS vs py-spy:"
echo "  - Shows KERNEL frames (CPU scheduler, syscalls, page faults)"
echo "  - Shows time spent in CUDA runtime (libcuda.so)"
echo "  - Useful when you suspect kernel-level overhead"
echo ""

PERF_FOLDED="$OUTDIR/${TARGET}_perf.folded"
PERF_SVG="$OUTDIR/${TARGET}_perf.svg"

if command -v perf &>/dev/null && [ -f "$FLAMEGRAPH_DIR/flamegraph.pl" ]; then
    
    # Step A: Record with perf
    # -g = capture call graphs (stack traces)
    # -F 99 = 99 samples per second (avoids exact multiples of timer freq)
    # -- = separator between perf args and application command
    echo "  Recording with perf (this runs the script)..."
    perf record -g -F 99 -- python $SCRIPT 2>/dev/null
    
    # Step B: Convert perf.data to folded stacks format
    # perf script outputs raw samples
    # stackcollapse-perf.pl folds them into "func1;func2;func3 count" format
    echo "  Collapsing stacks..."
    perf script | \
        "$FLAMEGRAPH_DIR/stackcollapse-perf.pl" \
        --no-inline \
        > "$PERF_FOLDED"
    
    # Step C: Generate SVG
    # --title = graph title
    # --width = SVG width in pixels
    # --bgcolor = background colour
    echo "  Generating SVG..."
    "$FLAMEGRAPH_DIR/flamegraph.pl" \
        --title "${TARGET}_training CPU Flamegraph" \
        --width 1400 \
        --bgcolor "#06080a" \
        "$PERF_FOLDED" \
        > "$PERF_SVG"
    
    echo "  ✓ perf flamegraph: $PERF_SVG"
    echo "     Open in browser: firefox $PERF_SVG"
    
    rm -f perf.data  # Clean up perf.data (can be large)
    
else
    echo "  Skipping: perf not found or FlameGraph not at $FLAMEGRAPH_DIR"
    echo "  Install FlameGraph: git clone https://github.com/brendangregg/FlameGraph ~/FlameGraph"
fi

echo ""
echo "============================================================"
echo "  FILES:"
ls -lh "$OUTDIR/${TARGET}"* 2>/dev/null || echo "  (no files generated)"
echo ""
echo "  NEXT STEPS:"
if [ "$TARGET" = "slow" ]; then
    echo "  1. Open $OUTDIR/slow_pyspy.svg in a browser"
    echo "  2. Find the widest bars — those are your bottlenecks"
    echo "  3. Run: bash profile_flamegraph.sh fast"
    echo "  4. Run: bash diff_flamegraph.sh"
elif [ "$TARGET" = "fast" ]; then
    echo "  1. Open $OUTDIR/fast_pyspy.svg in a browser"
    echo "  2. Compare width of the same functions vs slow version"
    echo "  3. Run: bash diff_flamegraph.sh"
fi
echo "============================================================"
