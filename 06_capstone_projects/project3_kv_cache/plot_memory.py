#!/usr/bin/env python3
"""
plot_memory.py  —  Project 3: Visualise GPU Memory Usage Over Time
====================================================================

HOW TO RUN:
    # After running memory_monitor.sh during a benchmark:
    python plot_memory.py

    # Compare two configs side-by-side:
    python plot_memory.py --file1 reports/constrained.csv --file2 reports/optimized.csv


"""

import argparse
import os
import sys

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("Install: pip install pandas matplotlib")
    sys.exit(1)

parser = argparse.ArgumentParser()
parser.add_argument("--file1",  default="reports/gpu_memory_log.csv",
                    help="Primary CSV file from memory_monitor.sh")
parser.add_argument("--file2",  default=None,
                    help="Optional second CSV for comparison")
parser.add_argument("--title",  default="GPU Memory & Utilisation During KV Cache Experiment")
parser.add_argument("--outfile",default="reports/memory_plot.png")
args = parser.parse_args()

def load_csv(path):
    """Load and validate memory log CSV."""
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run memory_monitor.sh first.")
        sys.exit(1)
    
    df = pd.read_csv(path, names=["t", "mem_used", "mem_free", "gpu_util", "temp"],
                     skiprows=1)
    
    # Clean up any whitespace in numeric columns
    for col in ["mem_used", "mem_free", "gpu_util", "temp"]:
        df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")
    
    df["mem_total"] = df["mem_used"] + df["mem_free"]
    df["mem_pct"]   = df["mem_used"] / df["mem_total"] * 100
    
    # Reset time to start at 0
    df["t"] = df["t"] - df["t"].iloc[0]
    
    return df

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"Loading {args.file1}...")
df1 = load_csv(args.file1)
df2 = load_csv(args.file2) if args.file2 else None

# ── Create figure ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle(args.title, fontsize=14, fontweight="bold")

# ── Colour palette ────────────────────────────────────────────────────────────
C1_MEM  = "#3fb950"   # green  — memory used
C1_FREE = "#21262d"   # dark   — memory free (background)
C1_UTIL = "#58a6ff"   # blue   — GPU utilisation
C2_MEM  = "#f78166"   # orange — second config memory
C2_UTIL = "#e3b341"   # gold   — second config utilisation

# ── Plot 1: Memory Usage ──────────────────────────────────────────────────────
ax1 = axes[0]

# Total memory as reference line
total_mb = df1["mem_total"].median()
ax1.axhline(y=total_mb, color="#444", linestyle="--", linewidth=1,
            label=f"Total VRAM ({total_mb:.0f} MB)")

# Memory used (filled area under curve)
ax1.fill_between(df1["t"], df1["mem_used"], alpha=0.3, color=C1_MEM)
ax1.plot(df1["t"], df1["mem_used"], color=C1_MEM, linewidth=1.5,
         label=f"Config 1: Memory Used")

if df2 is not None:
    ax1.fill_between(df2["t"], df2["mem_used"], alpha=0.2, color=C2_MEM)
    ax1.plot(df2["t"], df2["mem_used"], color=C2_MEM, linewidth=1.5,
             label=f"Config 2: Memory Used", linestyle="--")

# Warning line at 90% of total VRAM
ax1.axhline(y=total_mb * 0.90, color="#f78166", linestyle=":", linewidth=1,
            label="90% VRAM (danger zone)")

ax1.set_ylabel("GPU Memory (MB)", fontsize=11)
ax1.set_ylim(0, total_mb * 1.1)
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(True, alpha=0.2)
ax1.set_facecolor("#0d1117")
fig.patch.set_facecolor("#06080a")
ax1.tick_params(colors="#888")
ax1.yaxis.label.set_color("#888")
for spine in ax1.spines.values():
    spine.set_color("#21262d")

# Annotate peak memory
peak_mem  = df1["mem_used"].max()
peak_time = df1.loc[df1["mem_used"].idxmax(), "t"]
ax1.annotate(
    f"Peak: {peak_mem:.0f} MB\n({peak_mem/total_mb*100:.0f}% of VRAM)",
    xy=(peak_time, peak_mem),
    xytext=(peak_time + 2, peak_mem - total_mb * 0.15),
    arrowprops=dict(arrowstyle="->", color="#aaa"),
    color="#ccc", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#161b22", edgecolor="#3fb950", alpha=0.8)
)

# ── Plot 2: GPU Utilisation ───────────────────────────────────────────────────
ax2 = axes[1]

ax2.fill_between(df1["t"], df1["gpu_util"], alpha=0.3, color=C1_UTIL)
ax2.plot(df1["t"], df1["gpu_util"], color=C1_UTIL, linewidth=1.5,
         label="Config 1: GPU Utilisation")

if df2 is not None:
    ax2.fill_between(df2["t"], df2["gpu_util"], alpha=0.2, color=C2_UTIL)
    ax2.plot(df2["t"], df2["gpu_util"], color=C2_UTIL, linewidth=1.5,
             label="Config 2: GPU Utilisation", linestyle="--")

# Reference lines
ax2.axhline(y=90, color="#3fb950", linestyle=":", linewidth=1, alpha=0.5,
            label="90% target utilisation")
ax2.axhline(y=50, color="#f78166", linestyle=":", linewidth=1, alpha=0.5,
            label="50% warning threshold")

ax2.set_ylabel("GPU Utilisation (%)", fontsize=11)
ax2.set_xlabel("Time (seconds)", fontsize=11)
ax2.set_ylim(0, 105)
ax2.legend(loc="upper left", fontsize=9)
ax2.grid(True, alpha=0.2)
ax2.set_facecolor("#0d1117")
ax2.tick_params(colors="#888")
ax2.xaxis.label.set_color("#888")
ax2.yaxis.label.set_color("#888")
for spine in ax2.spines.values():
    spine.set_color("#21262d")

# ── Stats box ─────────────────────────────────────────────────────────────────
stats_text = (
    f"  Config 1 Stats:\n"
    f"  Peak VRAM:    {peak_mem:.0f} MB ({peak_mem/total_mb*100:.0f}%)\n"
    f"  Avg VRAM:     {df1['mem_used'].mean():.0f} MB\n"
    f"  Avg GPU util: {df1['gpu_util'].mean():.0f}%\n"
    f"  P95 GPU util: {df1['gpu_util'].quantile(0.95):.0f}%\n"
    f"  Duration:     {df1['t'].max():.0f}s"
)
ax2.text(
    0.99, 0.05, stats_text,
    transform=ax2.transAxes,
    fontsize=8, verticalalignment="bottom", horizontalalignment="right",
    fontfamily="monospace",
    bbox=dict(boxstyle="round", facecolor="#161b22", edgecolor="#21262d", alpha=0.8),
    color="#ccc"
)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(args.outfile, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
print(f"\n✓ Plot saved: {args.outfile}")

# Also print a text summary
print(f"\n  Memory Summary:")
print(f"  Total VRAM    : {total_mb:.0f} MB")
print(f"  Peak used     : {peak_mem:.0f} MB  ({peak_mem/total_mb*100:.0f}%)")
print(f"  Avg used      : {df1['mem_used'].mean():.0f} MB")
print(f"  Avg GPU util  : {df1['gpu_util'].mean():.0f}%")
print(f"\n  Add this plot to your portfolio write-up!")
