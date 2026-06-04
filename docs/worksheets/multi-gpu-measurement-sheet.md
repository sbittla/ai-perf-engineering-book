# Multi-GPU Scaling Measurement Sheet

A fillable companion to **Appendix D.5**. Run the protocol on a multi-GPU node and fill
the two sheets; the values drop directly into case study D.3 and the Chapter 29 capstone.

## Protocol
1. Run your training/inference script under `torchrun --nproc_per_node=N` for
   `N ∈ {1, 2, 4, 8}` (and 16 with two nodes). Record **steady-state** tokens/sec after warmup.
2. Profile one step at the largest N with `nsys profile` — confirm the NCCL collective
   overlaps the backward pass.
3. Measure raw collective bandwidth: `all_reduce_perf -b 1K -e 1G -f 2 -g <gpus>`.

## Sheet 1 — Scaling efficiency
`efficiency(N) = throughput(N) / (N × throughput(1))`. Ideal = 1.0; the gap is communication overhead.

| GPUs (N) | Throughput (tok/s) | Speedup vs 1 | Efficiency | Notes |
|---|---|---|---|---|
| 1 | | 1.00× | 1.00 | baseline |
| 2 | | | | |
| 4 | | | | |
| 8 | | | | |
| 16 | | | | crosses node boundary |

## Sheet 2 — Collective bandwidth (from nccl-tests)

| Message size | algbw (GB/s) | busbw (GB/s) | Notes |
|---|---|---|---|
| 1 KB | | | latency-bound |
| 1 MB | | | |
| 64 MB | | | |
| 1 GB | | | bandwidth-bound |

## How to read it
Efficiency should stay near 1.0 while collectives stay on NVLink and step down where they
first cross to inter-node InfiniBand — that step is the cliff. If busbw is far below the
link rate even for 1 GB messages, the collective is mistuned (algorithm, buffer size, or
topology), not the GPUs — tune it before adding nodes (Appendix E.3).
