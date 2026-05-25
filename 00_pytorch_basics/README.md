# 00 — PyTorch Basics

> **The exercises for Part I have moved.**
>
> All exercises are now in the [`I.Foundations/`](../I.Foundations/) directory,
> organised by chapter. See [`I.Foundations/README.md`](../I.Foundations/README.md)
> for the full exercise map and setup instructions.

## Exercise locations

| Chapter | Topic | File |
|---------|-------|------|
| 1 | Roofline model, MFU, economics | [`I.Foundations/1.What_Is_AI_Performance_Engineering/1.1_roofline_model.py`](../I.Foundations/1.What_Is_AI_Performance_Engineering/1.1_roofline_model.py) |
| 2 | Tensors | [`I.Foundations/2.PyTorch_Fundamentals/2.1_tensors.py`](../I.Foundations/2.PyTorch_Fundamentals/2.1_tensors.py) |
| 2 | Autograd | [`I.Foundations/2.PyTorch_Fundamentals/2.2_autograd.py`](../I.Foundations/2.PyTorch_Fundamentals/2.2_autograd.py) |
| 2 | nn.Module | [`I.Foundations/2.PyTorch_Fundamentals/2.3_nn_modules.py`](../I.Foundations/2.PyTorch_Fundamentals/2.3_nn_modules.py) |
| 2 | Training loop | [`I.Foundations/2.PyTorch_Fundamentals/2.4_training_loop.py`](../I.Foundations/2.PyTorch_Fundamentals/2.4_training_loop.py) |
| 2 | GPU timing & profiling | [`I.Foundations/2.PyTorch_Fundamentals/2.5_gpu_timing.py`](../I.Foundations/2.PyTorch_Fundamentals/2.5_gpu_timing.py) |
| 2 | Ten common mistakes | [`I.Foundations/2.PyTorch_Fundamentals/2.6_common_mistakes.py`](../I.Foundations/2.PyTorch_Fundamentals/2.6_common_mistakes.py) |
| 3 | CPU cache & PCIe | [`I.Foundations/3.The_AI_Hardware_Stack/3.1_cpu_and_memory.py`](../I.Foundations/3.The_AI_Hardware_Stack/3.1_cpu_and_memory.py) |
| 3 | HBM & Tensor Cores | [`I.Foundations/3.The_AI_Hardware_Stack/3.2_gpu_memory_and_compute.py`](../I.Foundations/3.The_AI_Hardware_Stack/3.2_gpu_memory_and_compute.py) |
| 3 | GPU spec reading | [`I.Foundations/3.The_AI_Hardware_Stack/3.3_hardware_survey.py`](../I.Foundations/3.The_AI_Hardware_Stack/3.3_hardware_survey.py) |

## Quick start

```bash
# Chapter 1
python "I.Foundations/1.What_Is_AI_Performance_Engineering/1.1_roofline_model.py"

# Chapter 2 (in order)
python "I.Foundations/2.PyTorch_Fundamentals/2.1_tensors.py"
python "I.Foundations/2.PyTorch_Fundamentals/2.2_autograd.py"
python "I.Foundations/2.PyTorch_Fundamentals/2.3_nn_modules.py"
python "I.Foundations/2.PyTorch_Fundamentals/2.4_training_loop.py"
python "I.Foundations/2.PyTorch_Fundamentals/2.5_gpu_timing.py"
python "I.Foundations/2.PyTorch_Fundamentals/2.6_common_mistakes.py"

# Chapter 3
python "I.Foundations/3.The_AI_Hardware_Stack/3.1_cpu_and_memory.py"
python "I.Foundations/3.The_AI_Hardware_Stack/3.2_gpu_memory_and_compute.py"
python "I.Foundations/3.The_AI_Hardware_Stack/3.3_hardware_survey.py"
```
