#!/usr/bin/env bash
# =============================================================================
# diff_flamegraph.sh  —  Project 4: Differential Flamegraph (Before vs After)
# =============================================================================
# PURPOSE:
#   Generate a DIFFERENTIAL flamegraph that visually highlights the performance
#   changes between slow_training.py and fast_training.py.
#
# DIFFERENTIAL FLAMEGRAPH COLOURS:
#   RED   = this code path got SLOWER after the change (appeared more often)
#           or is ONLY in the after profile (new hot path introduced)
#   BLUE  = this code path got FASTER (appeared less often in profile)
#           This is what you WANT to see where you applied fixes
#   Width = absolute amount of change
#
# WHAT YOU SHOULD SEE:
#   Blue over:
#     transforms.ToPILImage     → CPU augmentation removed
#     transforms.ColorJitter    → CPU augmentation removed
#     loss.item()               → GPU sync removed from hot loop
#   No significant red areas → no regressions introduced
#
# PORTFOLIO ARTIFACT:
#   Save diff.svg — it's the clearest visual proof of your optimisation.
#   One image shows exactly what you removed and how much time was saved.
#
# PRE-REQUISITES:
#   Run profile_flamegraph.sh slow  first
#   Run profile_flamegraph.sh fast  first
#   Then run this script.
# =============================================================================

set -euo pipefail

FLAMEGRAPH_DIR="${FLAMEGRAPH_DIR:-$HOME/FlameGraph}"
OUTDIR="reports"

echo "============================================================"
echo "  Differential Flamegraph: slow vs fast"
echo "============================================================"

# Check prereqs
if [ ! -f "$FLAMEGRAPH_DIR/difffolded.pl" ]; then
    echo "ERROR: FlameGraph scripts not found at $FLAMEGRAPH_DIR"
    echo "Install: git clone https://github.com/brendangregg/FlameGraph ~/FlameGraph"
    exit 1
fi

SLOW_FOLDED="$OUTDIR/slow_pyspy.folded"
FAST_FOLDED="$OUTDIR/fast_pyspy.folded"

if [ ! -f "$SLOW_FOLDED" ] || [ ! -f "$FAST_FOLDED" ]; then
    echo "ERROR: Missing folded stack files."
    echo "Run first:"
    echo "  bash profile_flamegraph.sh slow"
    echo "  bash profile_flamegraph.sh fast"
    exit 1
fi

echo ""
echo "Input files:"
echo "  Before (slow): $SLOW_FOLDED  ($(wc -l < "$SLOW_FOLDED") stack samples)"
echo "  After  (fast): $FAST_FOLDED  ($(wc -l < "$FAST_FOLDED") stack samples)"

# =============================================================================
# STEP 1: Generate Differential Flamegraph
# =============================================================================
echo ""
echo "[1/3] Generating differential flamegraph..."

DIFF_FOLDED="$OUTDIR/diff.folded"
DIFF_SVG="$OUTDIR/diff_flamegraph.svg"

# difffolded.pl <before.folded> <after.folded>
# Outputs a folded file where each stack's count is:
#   (count_in_after - count_in_before)
#
# Positive count → more time in after (worse or new code path): RED
# Negative count → less time in after (improved code path):     BLUE
#
# The -n flag normalises counts so profiles of different durations can compare
"$FLAMEGRAPH_DIR/difffolded.pl" \
    -n \
    "$SLOW_FOLDED" \
    "$FAST_FOLDED" \
    > "$DIFF_FOLDED"

# Generate the SVG from the differential folded file
"$FLAMEGRAPH_DIR/flamegraph.pl" \
    --title "Differential: fast_training vs slow_training (Blue=Faster, Red=Slower)" \
    --subtitle "Negative (blue) = time removed. Positive (red) = time added." \
    --width 1400 \
    --colors hot \
    "$DIFF_FOLDED" \
    > "$DIFF_SVG"

echo "  ✓ Differential flamegraph: $DIFF_SVG"
echo "     Open in browser: firefox $DIFF_SVG"

# =============================================================================
# STEP 2: Side-by-side comparison flamegraphs (same scale)
# =============================================================================
echo ""
echo "[2/3] Generating side-by-side comparison flamegraphs..."
echo "      (Normalised to same total sample count for fair comparison)"

# Normalise both profiles to the same total so widths are comparable
SLOW_NORMALISED="$OUTDIR/slow_normalised.folded"
FAST_NORMALISED="$OUTDIR/fast_normalised.folded"

# Count total samples in each
SLOW_TOTAL=$(awk '{sum+=$NF} END{print sum}' "$SLOW_FOLDED")
FAST_TOTAL=$(awk '{sum+=$NF} END{print sum}' "$FAST_FOLDED")

echo "  Slow profile samples: $SLOW_TOTAL"
echo "  Fast profile samples: $FAST_TOTAL"

# Scale fast profile to match slow sample count for fair visual comparison
# If fast has fewer samples (faster execution), scale up proportionally
awk -v slow="$SLOW_TOTAL" -v fast="$FAST_TOTAL" \
    '{
        n = split($0, a, " ");
        count = a[n];
        stack = "";
        for (i=1; i<n; i++) stack = stack " " a[i];
        scaled_count = int(count * slow / fast + 0.5);
        print substr(stack, 2) " " scaled_count
    }' "$FAST_FOLDED" > "$FAST_NORMALISED"

# Generate comparison SVGs
"$FLAMEGRAPH_DIR/flamegraph.pl" \
    --title "SLOW training — CPU Time Distribution" \
    --width 1400 "$SLOW_FOLDED" > "$OUTDIR/compare_slow.svg"

"$FLAMEGRAPH_DIR/flamegraph.pl" \
    --title "FAST training — CPU Time Distribution (normalised to same scale)" \
    --width 1400 "$FAST_NORMALISED" > "$OUTDIR/compare_fast.svg"

echo "  ✓ Slow flamegraph: $OUTDIR/compare_slow.svg"
echo "  ✓ Fast flamegraph: $OUTDIR/compare_fast.svg"

# =============================================================================
# STEP 3: Text summary of what changed
# =============================================================================
echo ""
echo "[3/3] Text analysis of removed code paths..."
echo ""

# Show which stacks shrank the most (blue areas)
echo "  TOP CODE PATHS THAT GOT FASTER (removed time — blue in diff):"
echo "  (sorted by time removed)"
echo ""
"$FLAMEGRAPH_DIR/difffolded.pl" -n "$SLOW_FOLDED" "$FAST_FOLDED" 2>/dev/null | \
    awk '{
        n = split($0, a, " ");
        count = a[n]+0;
        stack = "";
        for (i=1; i<n; i++) stack = stack " " a[i];
        if (count < 0) print (-count) " " substr(stack, 2)
    }' | \
    sort -rn | \
    head -10 | \
    awk '{printf "  -%6d samples | %s\n", $1, substr($0, length($1)+2)}' \
    2>/dev/null || echo "  (analysis requires difffolded.pl)"

echo ""
echo "  TOP CODE PATHS THAT GOT SLOWER (added time — red in diff):"
"$FLAMEGRAPH_DIR/difffolded.pl" -n "$SLOW_FOLDED" "$FAST_FOLDED" 2>/dev/null | \
    awk '{
        n = split($0, a, " ");
        count = a[n]+0;
        stack = "";
        for (i=1; i<n; i++) stack = stack " " a[i];
        if (count > 0) print count " " substr(stack, 2)
    }' | \
    sort -rn | \
    head -5 | \
    awk '{printf "  +%6d samples | %s\n", $1, substr($0, length($1)+2)}' \
    2>/dev/null || echo "  (none — good!)"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "  DIFFERENTIAL FLAMEGRAPH COMPLETE"
echo ""
echo "  FILES:"
ls -lh "$OUTDIR"/*.svg 2>/dev/null
echo ""
echo "  HOW TO SHARE AS A PORTFOLIO ARTIFACT:"
echo "  1. Open $DIFF_SVG in Chrome"
echo "  2. Screenshot the largest blue areas (your wins)"
echo "  3. Annotate with: 'Removed CPU augmentation: saved Xms/step'"
echo "  4. Add to GitHub README or blog post with before/after numbers"
echo "============================================================"
