"""
Exercise 0.1 — GPU Evolution: Where Does Your Hardware Sit in History?

Chapter 0: From Pixels to Petaflops — The GPU Revolution
Book: AI Systems Performance Engineering

Run:
    python 0.1_gpu_evolution.py

What this exercise does:
  1. Detects your GPU (or notes CPU-only) and reads its hardware properties.
  2. Prints a historical timeline table of key GPU generations.
  3. Locates your GPU on the timeline and shows the compute gap vs. historical chips.
  4. Calculates the theoretical speedup FP16/BF16 gives you over FP32 on your card.
  5. Computes your GPU's memory-bandwidth-to-compute ratio (b:f ratio) and explains
     whether your hardware is bandwidth-limited or compute-limited by design.

No TODOs here — this is a read-and-run exercise. Read each section's output,
understand what it means, then answer the reflection questions at the end.
"""

import sys
import math

# ── GPU history reference table ───────────────────────────────────────────────
# (year, name, arch, fp32_tflops, fp16_tflops, bw_gbs, vram_gb, notes)
GPU_HISTORY = [
    (2006, "GeForce 8800 GTX",  "Tesla",       0.5,    None,   86,    0.75,
     "First CUDA GPU. Birth of programmable GPU compute."),
    (2012, "GTX 580",           "Fermi",        1.6,    None,   192,   1.5,
     "AlexNet's GPU. Proved deep learning was practical."),
    (2014, "GTX 980",           "Maxwell",      4.6,    None,   224,   4.0,
     "First efficient consumer GPU for deep learning research."),
    (2016, "Tesla P100 (HBM2)", "Pascal",      10.6,   21.2,   732,   16.0,
     "First HBM memory, NVLink. Purpose-built AI data-centre card."),
    (2017, "Tesla V100 (HBM2)", "Volta",       14.0,   112.0,  900,   16.0,
     "Tensor Cores invented. 8× AI throughput leap over Pascal."),
    (2018, "RTX 2080 Ti",       "Turing",      13.4,   107.0,  616,   11.0,
     "First consumer Tensor Cores. INT8 inference hardware added."),
    (2020, "A100 80 GB",        "Ampere",      19.5,   312.0, 2000,   80.0,
     "BF16 Tensor Cores, MIG. Standard data-centre AI chip 2020–2023."),
    (2022, "RTX 4090",          "Ada Lovelace",82.6,   165.0, 1008,   24.0,
     "Consumer flagship. Best price/FLOP for independent researchers."),
    (2022, "H100 SXM",          "Hopper",      67.0,   989.0, 3350,   80.0,
     "Transformer Engine, FP8. Current data-centre standard."),
    (2024, "B200",              "Blackwell",   40.0,  4500.0, 8000,  192.0,
     "5th-gen Tensor Cores, FP4. 2024 frontier."),
]

COL = {
    "year": 6, "name": 20, "arch": 14,
    "fp32": 10, "fp16": 10, "bw": 10, "vram": 8,
}


def _hr(char="─", width=90):
    print(char * width)


def _row(*vals, widths):
    parts = [str(v).ljust(w) for v, w in zip(vals, widths)]
    print("  " + "  ".join(parts))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Detect local GPU
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  EXERCISE 0.1  —  GPU Evolution: Where Does Your Hardware Sit?")
print("═" * 70)

print("\n── SECTION 1: Your Hardware ─────────────────────────────────────────")

try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False
    print("  torch not found — install PyTorch to see live GPU info.")
    print("  Reference tables will still be printed below.\n")

local_name      = "CPU-only"
local_fp32      = None
local_bw        = None
local_vram      = None
local_sm_count  = None
local_cuda_ver  = None
local_cc        = None

if has_torch and torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    local_name     = props.name
    local_vram     = props.total_memory / 1e9
    local_sm_count = props.multi_processor_count
    local_cc       = f"{props.major}.{props.minor}"
    local_cuda_ver = torch.version.cuda

    # Rough FP32 TFLOP/s estimate: SM_count × cores_per_SM × 2 (FMA) × boost_clock
    # We use a conservative 1.5 GHz boost as a fallback; actual varies by SKU.
    # torch doesn't expose clock speed — user sees this as an approximation.
    cores_per_sm_map = {
        (3, 0): 192, (3, 5): 192,          # Kepler
        (5, 0): 128, (5, 2): 128,          # Maxwell
        (6, 0):  64, (6, 1): 128,          # Pascal
        (7, 0):  64,                        # Volta
        (7, 5):  64,                        # Turing
        (8, 0):  64, (8, 6):  128,         # Ampere
        (8, 9): 128,                        # Ada Lovelace
        (9, 0): 128,                        # Hopper
    }
    cores_per_sm = cores_per_sm_map.get(
        (props.major, props.minor), 128)
    total_cores = local_sm_count * cores_per_sm
    # Approximate: use 1500 MHz as a conservative estimate
    approx_ghz   = 1.5
    local_fp32 = round(total_cores * 2 * approx_ghz / 1e3, 1)  # TFLOP/s

    print(f"  GPU detected   : {local_name}")
    print(f"  Compute cap.   : {local_cc}  (CUDA {local_cuda_ver})")
    print(f"  SMs            : {local_sm_count}")
    print(f"  VRAM           : {local_vram:.1f} GB")
    print(f"  FP32 estimate  : ~{local_fp32} TFLOP/s  (≈1.5 GHz boost assumed)")
    print()
    print("  NOTE: torch does not expose clock speed. Check nvidia-smi or the")
    print("  manufacturer's spec sheet for your exact boost clock and TFLOP/s.")

else:
    if has_torch:
        print("  No CUDA GPU detected. Running on CPU.")
    print("  Showing reference data only.\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Historical timeline
# ══════════════════════════════════════════════════════════════════════════════

print("\n── SECTION 2: GPU Generation Timeline ───────────────────────────────")
print()
widths = [6, 20, 14, 10, 10, 10, 8]
_row("Year", "GPU", "Architecture", "FP32 TF", "FP16 TF", "BW GB/s", "VRAM",
     widths=widths)
_hr()

for (year, name, arch, fp32, fp16, bw, vram, note) in GPU_HISTORY:
    fp16_str = f"{fp16:.0f}" if fp16 else "—"
    _row(year, name, arch,
         f"{fp32}", fp16_str, f"{bw}", f"{vram} GB",
         widths=widths)

_hr()
print()
print("  FP16 TF = Tensor Core FP16 TFLOP/s (only available from Pascal onward).")
print("  BW = HBM/GDDR memory bandwidth in GB/s.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Where your GPU sits in history
# ══════════════════════════════════════════════════════════════════════════════

print("\n── SECTION 3: Your GPU on the Timeline ──────────────────────────────")

# Find the closest historical GPU by FP32 TFLOP/s
if local_fp32:
    closest = min(GPU_HISTORY, key=lambda r: abs(r[2+1] - local_fp32))  # fp32 index=2
    # Actually index: (year=0, name=1, arch=2, fp32=3, fp16=4, bw=5, vram=6)
    closest = min(GPU_HISTORY, key=lambda r: abs(r[3] - local_fp32))

    alexnet_fp32 = GPU_HISTORY[1][3]   # GTX 580
    v100_fp32    = GPU_HISTORY[4][3]   # V100
    a100_fp16    = GPU_HISTORY[6][4]   # A100 FP16
    h100_fp16    = GPU_HISTORY[8][4]   # H100 FP16

    print(f"\n  Your GPU ({local_name}) is closest to:")
    print(f"    {closest[1]} ({closest[0]}, {closest[2]} arch)")
    print(f"    Reference FP32: {closest[3]} TFLOP/s")
    print()

    # Speedup vs AlexNet's GPU
    speedup_alexnet = local_fp32 / alexnet_fp32
    print(f"  vs AlexNet's GTX 580 (2012):")
    print(f"    Your FP32: {local_fp32} TFLOP/s  |  GTX 580: {alexnet_fp32} TFLOP/s")
    print(f"    Speedup: {speedup_alexnet:.0f}× more raw FP32 compute")
    print(f"    → AlexNet trained in ~6 days on 2× GTX 580s.")
    days_equiv = 6 * 2 / speedup_alexnet
    print(f"    → On your single GPU, the same training would take "
          f"~{days_equiv*24:.1f} hours (FP32, no other optimizations).")
    print()

    # Speedup vs V100 baseline (Tensor Core era start)
    if local_fp32 < v100_fp32:
        print(f"  vs V100 (2017, first Tensor Core GPU):")
        print(f"    The V100 has {v100_fp32/local_fp32:.1f}× more FP32 throughput than your GPU.")
        print(f"    (But FP16/BF16 gap depends on whether your GPU has Tensor Cores.)")
    else:
        ratio = local_fp32 / v100_fp32
        print(f"  vs V100 (2017): your GPU has {ratio:.1f}× the FP32 throughput of a V100.")

else:
    print("\n  (No live GPU detected — install CUDA and torch to see placement.)")
    print()
    print("  Reference: AlexNet (2012) ran on GTX 580s at 1.6 TFLOP/s FP32.")
    print("  A modern A100 provides 312 TFLOP/s FP16 — ~195× more AI compute.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Precision speedup ratio on your GPU
# ══════════════════════════════════════════════════════════════════════════════

print("\n── SECTION 4: Precision Speedup on Your GPU ─────────────────────────")

if has_torch and torch.cuda.is_available():
    cc_major = torch.cuda.get_device_properties(0).major

    has_tensor_cores = cc_major >= 7   # Volta+
    has_bf16         = cc_major >= 8   # Ampere+
    has_fp8          = cc_major >= 9   # Hopper+

    print(f"\n  Compute capability {local_cc}:")
    print(f"    Tensor Cores (FP16 matmul speedup)  : {'YES ✓' if has_tensor_cores else 'NO  ✗'}")
    print(f"    BF16 Tensor Cores (AMP preferred)   : {'YES ✓' if has_bf16 else 'NO  ✗'}")
    print(f"    FP8 Tensor Cores (Hopper+)          : {'YES ✓' if has_fp8 else 'NO  ✗'}")

    if has_tensor_cores:
        print()
        print("  Tensor Cores are ACTIVE on your GPU. To use them:")
        print("    - Run matrix operations with dimensions that are multiples of 8")
        print("    - Use torch.amp.autocast('cuda') or model.to(torch.bfloat16)")
        print()
        print("  Rough speedup from FP32 → FP16/BF16 on Tensor Core hardware:")
        print("    Typical measured: 2–8× for GEMM-heavy models")
        print("    Theoretical (Tensor Core spec): 8–16× for pure matmul")
        print("    You will measure the real number in Exercise 6.1.")
    else:
        print()
        print("  No Tensor Cores detected (pre-Volta GPU).")
        print("  FP16 gives a minor speedup (~1.5×) from reduced memory bandwidth.")
        print("  Consider upgrading to an RTX 30xx or newer for Tensor Core benefit.")
else:
    print("\n  (No live GPU — showing typical values by generation)")
    print()
    tbl = [
        ("Pre-Volta (GTX 10xx and older)", "No Tensor Cores", "~1.5×",    "Memory BW reduction only"),
        ("Turing (RTX 20xx)",              "FP16",            "~4–6×",    "1st consumer Tensor Cores"),
        ("Ampere (RTX 30xx, A100)",        "FP16 + BF16",     "~6–8×",    "BF16 avoids loss scaling"),
        ("Ada (RTX 40xx)",                 "FP16 + BF16",     "~6–8×",    "4th-gen Tensor Cores"),
        ("Hopper (H100)",                  "FP16/BF16/FP8",   "~8–16×",   "Transformer Engine"),
    ]
    for row in tbl:
        print(f"    {row[0]:<32} {row[1]:<16} {row[2]:<8} {row[3]}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Bandwidth-to-compute ratio (b:f ratio)
# ══════════════════════════════════════════════════════════════════════════════

print("\n── SECTION 5: Bandwidth-to-Compute Balance ──────────────────────────")
print()
print("  The b:f ratio = memory bandwidth (GB/s) / AI peak FLOP/s (TFLOP/s)")
print("  It tells you how many bytes of memory each TFLOP of compute can afford.")
print("  Lower b:f means the GPU is more compute-heavy; bandwidth becomes the limit.")
print()

ref_gpus = [
    ("GTX 580 (2012)",   192,    1.6,    None),
    ("V100 (2017)",      900,   14.0,   112.0),
    ("A100 (2020)",     2000,   19.5,   312.0),
    ("H100 (2022)",     3350,   67.0,   989.0),
    ("B200 (2024)",     8000,   40.0,  4500.0),
]

print(f"  {'GPU':<22} {'BW GB/s':>9} {'AI TFLOP/s':>12} {'b:f (bytes/FLOP)':>18}")
_hr(width=65)
for name, bw, fp32, fp16 in ref_gpus:
    ai_flops = fp16 if fp16 else fp32
    bf_ratio = bw / (ai_flops * 1000)   # bytes per FLOP
    print(f"  {name:<22} {bw:>9,} {ai_flops:>12.0f} {bf_ratio:>18.4f}")

if local_fp32 and local_vram:
    # We don't know local BW precisely; we can compute from vram if GDDR6/HBM
    # Just note that we can't without specs
    print()
    print(f"  Your GPU ({local_name}):")
    print(f"    FP32 estimate: ~{local_fp32} TFLOP/s  |  VRAM: {local_vram:.0f} GB")
    print(f"    For an exact b:f ratio, find your GPU's bandwidth spec and compute:")
    print(f"    b:f = bandwidth_GB_per_s / (fp16_tflops × 1000)")

print()
print("  Interpreting b:f ratio:")
print("    > 0.20  bytes/FLOP → bandwidth-rich GPU (older, fewer Tensor Cores)")
print("    0.05–0.20           → balanced; workload type determines bottleneck")
print("    < 0.05  bytes/FLOP → compute-rich GPU; most workloads are memory-bound")
print()
print("  The H100 (0.0034 bytes/FLOP) is so compute-heavy that nearly every")
print("  AI workload runs into the memory wall before the compute ceiling.")
print("  This is why quantization and FlashAttention matter more, not less,")
print("  on newer hardware.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Reflection questions
# ══════════════════════════════════════════════════════════════════════════════

print("\n── SECTION 6: Reflection Questions ──────────────────────────────────")
print("""
  Think through these before moving to Chapter 1. No code needed — just reasoning.

  Q1. AlexNet trained for 6 days on two GTX 580s in 2012.
      If it ran today on two H100s, how long would it take?
      (Hint: compare FP16 TFLOP/s, then consider that AlexNet used FP32.)

  Q2. The H100 has 989 FP16 TFLOP/s but only 3,350 GB/s of bandwidth.
      The ridge point is 989,000 / 3,350 ≈ 295 FLOPs/byte.
      Most LLM decode operations have arithmetic intensity < 5 FLOPs/byte.
      What fraction of the H100's compute is LLM decode actually using?

  Q3. A colleague says: "We should upgrade from A100 to H100 — it's 3× faster."
      Under what conditions is this true? Under what conditions is it false?
      (Hint: think about which ceiling your workload is currently hitting.)

  Q4. Your model trains in FP32 and you are about to switch to BF16.
      Your GPU is an RTX 3090 (Ampere, compute capability 8.6).
      What speedup do you expect for: (a) a large GEMM, (b) a ReLU activation?
      Why are the answers different?
""")

print("═" * 70)
print("  Exercise 0.1 complete. Proceed to Chapter 1: The Roofline Model.")
print("═" * 70 + "\n")
