# Part VII — The 2024–2026 Hardware Landscape

This part covers the current generation of AI accelerators beyond the H100/A100 baseline used in Parts I–VI. All exercises run on CPU-only machines using reference data tables; no physical access to Blackwell, MI300X, or Gaudi hardware is required.

## Chapters

| Chapter | Topic | Exercises |
|---------|-------|-----------|
| 20 | NVIDIA Blackwell | `20.1_blackwell_architecture.py` · `20.2_blackwell_profiling.py` |
| 21 | AMD MI300X / ROCm | `21.1_roofline_multi_gpu.py` · `21.2_rocm_profiling.py` |
| 22 | Custom Silicon & CXL | `22.1_hardware_landscape.py` |

## Exercise Numbering

Unlike Parts I–VI, Part VII has no "background/intro" `.0_` file. Exercise numbers map 1:1 to manuscript section groups:

- `20.1` covers manuscript sections 20.1–20.4 (architecture, HBM3e, NVLink 5, Grace Blackwell)
- `20.2` covers manuscript sections 20.5–20.6 (RTX 5090, profiling adaptations)
- `21.1` covers manuscript sections 21.1 and 21.4 (MI300X architecture + hardware selection)
- `21.2` covers manuscript sections 21.2–21.3 (ROCm profiling, PyTorch migration)
- `22.1` covers manuscript sections 22.1–22.5 (all custom silicon and CXL)

## Run Order

```bash
python VII.Hardware_Landscape/20.NVIDIA_Blackwell/20.1_blackwell_architecture.py
python VII.Hardware_Landscape/20.NVIDIA_Blackwell/20.2_blackwell_profiling.py
python VII.Hardware_Landscape/21.AMD_MI300X_ROCm/21.1_roofline_multi_gpu.py
python VII.Hardware_Landscape/21.AMD_MI300X_ROCm/21.2_rocm_profiling.py
python VII.Hardware_Landscape/22.Custom_Silicon/22.1_hardware_landscape.py
```

## Prerequisites

- Python 3.10+
- `torch` (optional — exercises degrade gracefully to reference tables without it)
- No GPU required; no cloud credentials required
