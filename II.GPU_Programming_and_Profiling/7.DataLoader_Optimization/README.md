# Chapter 7 Exercises — DataLoader Optimization

Two exercises covering the most common source of GPU underutilisation in production training pipelines: a slow data pipeline.

| File | Section | Topics | Estimated Time |
|---|---|---|---|
| 7.1_dataloader_pipeline.py | 7.1 | num_workers, pin_memory, prefetch_factor, .item() anti-pattern | 60 min |
| 7.2_io_bottleneck.py | 7.2 | GPU starvation, idle_pct measurement, nvidia-smi, fix checklist | 60 min |

## Hardware note

Both exercises simulate I/O latency using `time.sleep()` and run on CPU or CUDA. The GPU idle time measurement in 7.2 is most informative with a CUDA GPU — on CPU the "compute" time is the model forward pass in Python, not a GPU kernel, so the idle fraction reflects Python overhead rather than true GPU starvation.

For the nvidia-smi diagnosis in Section 7.2.3, you need a system with a CUDA GPU and the nvidia-smi utility installed (standard on Linux with NVIDIA drivers).

Complete both exercises to finish all of Part II — GPU Programming & Profiling.
