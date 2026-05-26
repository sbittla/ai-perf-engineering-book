# Chapter 1 Exercises — What Is AI Performance Engineering?

One exercise covering the roofline model and the economic case for GPU performance engineering.

## exercise_01_1.1_roofline_model.py

**Book sections:** 1.1 (hardware stack), 1.2 (roofline model), 1.5 (economics), 1.6 (optimisation hierarchy)

**What you will do:**
1. Read your GPU's hardware specifications (peak FLOP/s, HBM bandwidth)
2. Compute the ridge point — the arithmetic intensity that separates memory-bound from compute-bound
3. Calculate arithmetic intensity for common operations (GEMM, elementwise, LLM decode)
4. Classify each operation as memory-bound or compute-bound
5. Measure actual throughput for a matrix multiply and compute MFU
6. Compare observed performance to the roofline prediction

**Estimated time:** 30 minutes

**Runs on:** CPU or GPU. GPU sections give the most informative output.
