# Book "Reference Result" Callouts (Draft)

Per-section callouts to drop into the manuscript, mapped to the **section titles** in
`…_KDP_MANUSCRIPT_v28-print.docx` (anchor by title, **not** by number — the book inserts
"Background" interludes so repo exercise numbers and book section numbers don't always match).

## How to use these

- **Render each as a single-row table** (1 column, header + one body cell). This matches the
  book's existing convention (the manuscript already uses 318 tables) and is KDP-print-safe — no
  new paragraph style needed. Optionally apply a light cell shade.
- **Insert at the subsection** named in each "📍 Placement" line.
- Language is deliberately **rounded / relative** so it won't date. Keep it that way; exact
  figures live in [`REFERENCE_RESULTS.md`](REFERENCE_RESULTS.md).
- **State the disclaimer once, globally** (see below) instead of repeating it in every box.
- Numbers below are from the reference setup (RTX 4060, PyTorch 2.6). Do an editorial pass for
  voice before inserting.

### One-time global disclaimer (put in the Preface / "How to Use the Companion Repository")

> **Reading "Reference Result" boxes.** Throughout the book, boxed *Reference Results* show
> measurements from a reference setup (NVIDIA RTX 4060, PyTorch 2.6). Treat the **ratios and the
> direction** as the lesson — your absolute numbers will differ with GPU, driver, and framework
> version, often dramatically on data-center hardware. Live, regenerated results and full logs are
> in the companion repository (github.com/sbittla/ai-perf-engineering-book).

---

## Callout 1 — Chapter 3

📍 **Placement:** Ch 3 → §3.5 *Reading GPU Specifications for Performance Engineers* → after
"TFLOP/s: Theoretical Peak vs Sustained".

| Reference Result — Roofline ridge point |
|---|
| Reading the GPU's own specs gives you a roofline. On the reference laptop GPU, peak FP16 throughput is ~50 TFLOP/s against ~270 GB/s of memory bandwidth, putting the ridge point near ~190 FLOPs/byte. Any kernel below that intensity is memory-bound — which, as later chapters show, is most of LLM decoding. |

---

## Callout 2 — Chapter 4 (occupancy)

📍 **Placement:** Ch 4 → §4.1 *The GPU Thread Hierarchy* → after "Why Batch Size Matters for
Occupancy".

| Reference Result — Batching fills the GPU |
|---|
| Increasing batch size lifts achieved matmul throughput by roughly 2× as otherwise-idle SMs fill up, with the occupancy "knee" (~80% of peak) arriving at a small batch size. Below the knee you are occupancy-limited; above it, compute-bound. On bigger GPUs the climb is steeper and the knee comes later. A branchy, two-path kernel also ran ~10× slower than a uniform one here — warp divergence is real. |

---

## Callout 3 — Chapter 4 (Tensor Cores & fusion)

📍 **Placement:** Ch 4 → §4.3 *Tensor Cores and Kernel Fusion* → after "BF16 vs FP16: Which to
Use" (and "Kernel Fusion: Why It Helps").

| Reference Result — Tensor Cores & fusion |
|---|
| BF16/FP16 matmuls on Tensor Cores ran about 2× faster than FP32 here (and far more on data-center GPUs). One counter-intuitive finding: `torch.compile` was *slower* than eager on a tiny compute-bound block, because kernel-launch overhead outweighed the fusion benefit. Fusion pays off on larger, memory-bound graphs — match the tool to the workload. |

---

## Callout 4 — Chapter 6

📍 **Placement:** Ch 6 → §6.1 *Numeric Precision and Automatic Mixed Precision* → after
"torch.autocast: One Line to Mixed Precision".

| Reference Result — autocast is a speed win, not a memory win |
|---|
| Wrapping the forward pass in `torch.autocast` made it ~1.5× faster by routing matmuls through Tensor Cores. Note what it did **not** do: peak memory stayed about the same. In a forward pass, autocast keeps FP32 weights and adds FP16 casts, so it trades a little memory for speed. Real memory savings come from storing the model in half precision, not from autocast. |

---

## Callout 5 — Chapter 7 (pipeline tuning)

📍 **Placement:** Ch 7 → §7.1 *DataLoader Pipeline Tuning* → after "The .item() Anti-Pattern"
(num_workers result can sit near "num_workers: The Most Important Parameter").

| Reference Result — cheap DataLoader wins |
|---|
| Two of the easiest training speedups: moving data loading to background workers was ~1.5–2× faster than single-threaded loading, and removing a per-step `.item()` call (which forces a CPU-to-GPU sync every iteration) was ~4× faster. Both are one-line changes. |

---

## Callout 6 — Chapter 7 (I/O bottleneck)

📍 **Placement:** Ch 7 → §7.2 *Diagnosing and Fixing I/O Bottlenecks* → after "Measuring
idle_pct".

| Reference Result — a starved GPU is mostly idle |
|---|
| With a slow data source, the GPU sat **>90% idle**, waiting on the CPU to deliver batches; with the same model fed from memory, idle time dropped to ~15%. `idle_pct` is the single number that tells you whether to optimize the input pipeline or the kernels. |

---

## Callout 7 — Chapter 9

📍 **Placement:** Ch 9 → §9.1 *The CPU Memory Hierarchy* → after "Cache Lines and Sequential
Access" / "Matrix Layout and PyTorch".

| Reference Result — layout dominates memory throughput |
|---|
| Reading a large matrix sequentially (cache-friendly) versus through a strided column gather (cache-unfriendly) differed by **more than 15×** on the reference machine — the same penalty `.contiguous()` exists to avoid. A measurement caveat worth teaching: NumPy's `sum(axis=…)` will *not* reveal this, because its reduction reads memory in cache-friendly blocks regardless of axis; you have to force the strided access to see the cost. |

---

## Callout 8 — Chapter 10

📍 **Placement:** Ch 10 → §10.1 *Prefill vs Decode Phases* → after "The Prefill Phase"
(cross-reference the TTFT definition in §10.3 *Inference Metrics*).

| Reference Result — TTFT grows with prompt length |
|---|
| Time-To-First-Token rises with prompt length, because prefill is compute-bound and processes the whole prompt in one pass. The signal is small for short prompts — TTFT there is dominated by fixed kernel-launch overhead — so you must average several runs to see it cleanly. That itself is a lesson in measuring sub-millisecond GPU work. |

---

**Optional status line** (Preface or back matter):
> *All 77 exercises in the companion repository were verified passing on the reference setup; the
> repository includes an auto-generated execution report and full per-exercise logs.*
