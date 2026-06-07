# Chapter 13 — Distributed Inference

Four exercises covering the parallelism strategies used when a model exceeds single-GPU VRAM or when a single GPU is too slow.

## Exercises

| File | Topic | Key Skill |
|------|-------|-----------|
| `13.1_tensor_parallelism.py` | Weight sharding, AllReduce cost, compute/comm ratio | Model TP memory; compute/comm roofline; choose TP degree |
| `13.2_nccl_collectives.py` | AllReduce, AllGather, ring bandwidth model | Implement collective time estimators; compare NVLink vs PCIe |
| `13.3_fsdp_and_pipeline.py` | FSDP memory, pipeline bubble, 3D parallelism | Compute bubble fraction; model 3D memory; choose strategy |
| `13.4_distributed_simulation.py` | Single-node simulation of TP AllReduce, NCCL timing, and timeout failure modes via PyTorch CPU process groups (`torchrun --nproc_per_node=2`) | Experience real collective semantics without a cluster |

## Decision Guide

| Strategy | Shard what | Communication | Use when |
|----------|-----------|---------------|----------|
| **Tensor Parallel** | Weight matrices | AllReduce per layer | Within node (NVLink), latency-critical |
| **Pipeline Parallel** | Layer groups | Activation passing | Across nodes, throughput-oriented |
| **FSDP** | Params + gradients + optimizer | AllGather + ReduceScatter | Training/fine-tuning, memory-constrained |
| **Data Parallel** | Nothing (replicas) | Gradient AllReduce | Multiple copies, large datasets |

## Run Order

```bash
python 13.1_tensor_parallelism.py   # weight sharding and AllReduce cost
python 13.2_nccl_collectives.py     # collective communication deep dive
python 13.3_fsdp_and_pipeline.py    # FSDP, pipeline bubble, 3D parallelism

# Single-node distributed simulation (2 processes, CPU-only OK):
torchrun --nproc_per_node=2 13.4_distributed_simulation.py
```

## Quick Reference

```bash
# vLLM with Tensor Parallelism
vllm serve meta-llama/Llama-2-70b-chat-hf --tensor-parallel-size 4

# HuggingFace Accelerate (automatic device_map)
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Megatron-LM 3D parallel launch
torchrun --nproc_per_node 8 pretrain_gpt.py \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 4
```
