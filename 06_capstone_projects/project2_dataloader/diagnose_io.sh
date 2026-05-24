#!/usr/bin/env bash
# =============================================================================
# diagnose_io.sh  —  Project 2, Step 2: Diagnose with Linux Perf Tools
# =============================================================================
# PURPOSE:
#   While slow_dataloader.py runs in another terminal, use these commands
#   to observe and diagnose the bottleneck from the outside — no code changes.
#
#   This is exactly what you'd do on a production system where you can't
#   easily modify the application code.
#
# HOW TO USE:
#   Terminal 1:  python slow_dataloader.py
#   Terminal 2:  bash diagnose_io.sh
#
# EACH SECTION runs independently. Run them one at a time in order,
# or run individual commands directly.
# =============================================================================

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info() { echo -e "\n${GREEN}══════════════════════════════════════════════${NC}"; 
         echo -e "${GREEN}  $*${NC}";
         echo -e "${GREEN}══════════════════════════════════════════════${NC}\n"; }
note() { echo -e "${YELLOW}  ► $*${NC}"; }

PYTHON_PID=$(pgrep -f "slow_dataloader.py" 2>/dev/null | head -1 || echo "")

# =============================================================================
# STEP 1 — GPU utilisation monitor (should show low, fluctuating %)
# =============================================================================
info "STEP 1: GPU Utilisation (10 seconds)"
note "Healthy GPU: ~90-100% steady"
note "Bottlenecked GPU: low %, fluctuating — gaps while CPU loads data"
note ""
note "COLUMNS:"
note "  timestamp  — time of measurement"
note "  name       — GPU name"
note "  utilization.gpu — % of time GPU had active warps (0-100)"
note "  memory.used — GPU VRAM consumed"
echo ""

# -l 1 = refresh every 1 second
# --format=csv,noheader,nounits = machine-readable output
nvidia-smi \
    --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.free \
    --format=csv,noheader \
    -l 1 \
    --loop-ms=500 &   # background so we can run other commands too

GPU_MON_PID=$!
sleep 10
kill $GPU_MON_PID 2>/dev/null || true
echo ""
note "If GPU utilisation was low and fluctuating → GPU is being starved by CPU"

# =============================================================================
# STEP 2 — vmstat: CPU states and memory pressure
# =============================================================================
info "STEP 2: vmstat — CPU States (10 seconds)"
note "COLUMNS TO WATCH:"
note "  us = user CPU (application code)"
note "  sy = system CPU (kernel, I/O)"
note "  wa = iowait — CPU% waiting for I/O to complete (BAD if high)"
note "  id = idle (GPU training should keep CPU busy, not idle)"
note "  si/so = swap in/out — if > 0, system is memory-starved"
echo ""

# 1 = print every 1 second
# 10 = print 10 times then stop
vmstat 1 10

note ""
note "HIGH 'wa' (>20%) = disk I/O is the bottleneck"
note "HIGH 'id' (>50%) = CPU has nothing to do (GPU is the bottleneck?)"
note "HIGH 'us' (>80%) = CPU-bound processing (augmentation bottleneck)"

# =============================================================================
# STEP 3 — iostat: disk throughput and utilisation
# =============================================================================
info "STEP 3: iostat — Disk I/O (10 seconds)"
note "COLUMNS TO WATCH:"
note "  %util  — how busy the disk is (100% = saturated)"
note "  await  — average I/O wait time in milliseconds"
note "  r/s    — reads per second"
note "  rkB/s  — read throughput in KB/s"
echo ""

# -x = extended stats (includes %util, await)
# -z = suppress devices with zero activity
# 1  = refresh every 1 second
# 10 = 10 intervals then stop
iostat -xz 1 10

note ""
note "%util near 100% = disk is saturated (move data to SSD or RAM disk)"
note "await > 10ms     = disk seek latency is significant"

# =============================================================================
# STEP 4 — opensnoop (eBPF): trace file opens by the Python process
# =============================================================================
info "STEP 4: opensnoop — Tracing File Opens (5 seconds)"
note "Shows EVERY file the Python process opens, with latency"
note "Reveals: checkpoint loads, dataset file reads, hidden config reads"
echo ""

if command -v opensnoop-bpfcc &>/dev/null || command -v opensnoop &>/dev/null; then
    OPENSNOOP=$(command -v opensnoop-bpfcc 2>/dev/null || command -v opensnoop)
    
    if [ -n "$PYTHON_PID" ]; then
        note "Tracing PID $PYTHON_PID for 5 seconds..."
        # -p = filter to this PID only
        # timeout 5 = stop after 5 seconds
        sudo timeout 5 "$OPENSNOOP" -p "$PYTHON_PID" 2>/dev/null || \
            sudo timeout 5 "$OPENSNOOP" 2>/dev/null | grep python | head -30 || true
    else
        note "slow_dataloader.py not running. Showing all opens for 3s..."
        sudo timeout 3 "$OPENSNOOP" 2>/dev/null | head -20 || true
    fi
else
    note "opensnoop not found. Install: sudo apt install bpfcc-tools"
    note "Manual alternative using strace:"
    if [ -n "$PYTHON_PID" ]; then
        sudo strace -e trace=openat -p "$PYTHON_PID" 2>&1 | head -20 || true
    fi
fi

# =============================================================================
# STEP 5 — biolatency (eBPF): block I/O latency histogram
# =============================================================================
info "STEP 5: biolatency — Disk I/O Latency Distribution (5 seconds)"
note "Shows DISTRIBUTION of disk I/O latency as a histogram"
note "Reveals: whether latency is consistent (SSD) or variable (HDD)"
echo ""

if command -v biolatency-bpfcc &>/dev/null || command -v biolatency &>/dev/null; then
    BIOLATENCY=$(command -v biolatency-bpfcc 2>/dev/null || command -v biolatency)
    # -D = per-disk breakdown
    sudo timeout 5 "$BIOLATENCY" -D 2>/dev/null || true
else
    note "biolatency not found. Install: sudo apt install bpfcc-tools"
fi

# =============================================================================
# STEP 6 — nsys: GPU timeline capture
# =============================================================================
info "STEP 6: nsys Timeline — GPU Idle Gap Analysis"
note "Captures a 10-second window showing where the GPU is idle"
note "Open the .nsys-rep in nsys-ui to see the gaps between kernels"
echo ""

mkdir -p reports
if command -v nsys &>/dev/null; then
    note "Attaching nsys to slow_dataloader.py (10s snapshot)..."
    nsys profile \
        --delay=2 \
        --duration=10 \
        --trace=cuda,osrt \
        --output=reports/slow_dataloader_snapshot \
        python slow_dataloader.py --steps 30 2>/dev/null || true
    
    note "Saved: reports/slow_dataloader_snapshot.nsys-rep"
    note "Open in nsys-ui to see gaps in the GPU timeline row"
else
    note "nsys not found — install Nsight Systems"
fi

# =============================================================================
# STEP 7 — perf stat: cache misses during augmentation
# =============================================================================
info "STEP 7: perf stat — Hardware Counters"
note "Measures CPU cache miss rate during the training loop"
note "High cache-misses = CPU augmentation pipeline is causing cache thrashing"
echo ""

if command -v perf &>/dev/null; then
    # -e = specific events to measure
    # cache-misses = L1/L2/L3 cache misses (memory inefficiency)
    # instructions = total instructions retired
    # cycles        = CPU clock cycles elapsed
    # branch-misses = mispredicted branches (pipeline flushes)
    perf stat \
        -e cycles,instructions,cache-misses,cache-references,branch-misses \
        python slow_dataloader.py --steps 20 --sleep-ms 0 \
        2>&1 | tail -20
else
    note "perf not found. Install: sudo apt install linux-tools-generic"
fi

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo -e "${GREEN}"
echo "  ╔═══════════════════════════════════════════════════╗"
echo "  ║  DIAGNOSIS SUMMARY                                ║"
echo "  ╠═══════════════════════════════════════════════════╣"
echo "  ║  Signs of DataLoader bottleneck:                  ║"
echo "  ║  • nvidia-smi: GPU util < 60%, fluctuating        ║"
echo "  ║  • vmstat: 'wa' > 20% or 'id' > 30%              ║"
echo "  ║  • iostat: %util near 100% on data disk           ║"
echo "  ║  • nsys: visible gaps in CUDA timeline            ║"
echo "  ║                                                   ║"
echo "  ║  Next: run fast_dataloader.py to fix it           ║"
echo "  ╚═══════════════════════════════════════════════════╝"
echo -e "${NC}"
