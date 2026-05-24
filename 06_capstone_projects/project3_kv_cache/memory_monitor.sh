#!/usr/bin/env bash
# =============================================================================
# memory_monitor.sh  —  Project 3: GPU Memory Logging to CSV
# =============================================================================
# PURPOSE:
#   Continuously log GPU memory usage to a CSV file while the vLLM server
#   processes concurrent requests. The resulting data feeds plot_memory.py
#   to produce a visual graph of memory vs time.
#
# HOW TO USE:
#   Terminal 1: bash kv_pressure_server.sh           (vLLM server)
#   Terminal 2: bash memory_monitor.sh               (this script)
#   Terminal 3: python concurrent_requests.py ...    (load generator)
#   Terminal 4: python plot_memory.py                (after experiment ends)
#
# OUTPUT:
#   reports/gpu_memory_log.csv  — timestamped memory samples
# =============================================================================

OUTDIR="reports"
OUTFILE="$OUTDIR/gpu_memory_log.csv"
INTERVAL_MS=500   # Sample every 500ms

mkdir -p "$OUTDIR"

echo "GPU Memory Monitor"
echo "Output: $OUTFILE"
echo "Sampling every ${INTERVAL_MS}ms"
echo "Press Ctrl+C to stop"
echo ""

# Write CSV header
echo "timestamp_s,memory_used_mb,memory_free_mb,gpu_util_pct,temp_c" > "$OUTFILE"

# Record start time for relative timestamps
START=$(date +%s%N)  # nanoseconds

while true; do
    # Query GPU stats in CSV format without headers or units
    # --query-gpu selects which metrics to retrieve
    # --format=csv,noheader,nounits = clean output for parsing
    LINE=$(nvidia-smi \
        --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu \
        --format=csv,noheader,nounits \
        2>/dev/null | head -1)
    
    if [ -z "$LINE" ]; then
        echo "nvidia-smi not available"
        exit 1
    fi
    
    # Calculate elapsed seconds with decimal precision
    NOW=$(date +%s%N)
    ELAPSED_S=$(echo "scale=2; ($NOW - $START) / 1000000000" | bc)
    
    # Append to CSV: timestamp, memory_used, memory_free, gpu_util, temp
    echo "$ELAPSED_S,$LINE" >> "$OUTFILE"
    
    # Print live to terminal too
    USED=$(echo "$LINE" | cut -d',' -f1 | tr -d ' ')
    UTIL=$(echo "$LINE" | cut -d',' -f3 | tr -d ' ')
    printf "\r  t=%.1fs  VRAM: %s MB used   GPU: %s%%    " "$ELAPSED_S" "$USED" "$UTIL"
    
    # Sleep for interval (convert ms to seconds)
    sleep $(echo "scale=3; $INTERVAL_MS / 1000" | bc)
done
