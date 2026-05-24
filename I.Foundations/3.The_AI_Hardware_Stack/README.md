# Chapter 3 Exercises — The AI Hardware Stack

Three exercises covering CPU, PCIe, and GPU hardware internals.

| File | Sections | Topics | Time |
|---|---|---|---|
| exercise_01_cpu_and_memory.py | 3.1, 3.2 | Cache effects, DataLoader access patterns, PCIe bandwidth, pinned memory | 45 min |
| exercise_02_gpu_memory_and_compute.py | 3.3, 3.4 | HBM bandwidth, Tensor Core activation, alignment, kernel fusion | 60 min |
| exercise_03_hardware_survey.py | 3.5 | GPU specs, ridge point, MFU, thermal monitoring, multi-GPU topology | 45 min |

**Hardware requirements:** CUDA GPU required for most sections.
Sections skip gracefully on CPU-only machines with a descriptive message.

**Before Part II:** complete exercise_03_hardware_survey.py and record your
GPU's peak FLOP/s, HBM bandwidth, and ridge point — you will use them
throughout the profiling chapters.
