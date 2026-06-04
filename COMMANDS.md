# Profiling Commands Reference

Every command you need, grouped by tool and use case.

---

## GPU Timeline — Nsight Systems (`nsys`)

```bash
# Quick stats summary (no GUI needed)
nsys profile --stats=true python train.py

# Full timeline for GUI (open with nsys-ui)
nsys profile -o report --trace=cuda,nvtx,osrt python train.py

# Attach to running server after 5s warmup, capture 10s
nsys profile --delay=5 --duration=10 -o snap python serve.py

# Include GPU memory allocation events
nsys profile --cuda-memory-usage=true --trace=cuda,nvtx python infer.py

# Analyse saved report from CLI
nsys analyze report.nsys-rep --report gputrace

# Open GUI
nsys-ui report.nsys-rep
```

**What to look for:**
- White gaps in GPU row → CPU bottleneck (DataLoader, Python overhead)
- Long H2D bars → slow data transfer (fix: `pin_memory=True`)
- Many tiny kernels → kernel launch overhead (fix: `torch.compile`)

---

## Per-Kernel Analysis — Nsight Compute (`ncu`)

```bash
# Quick summary of all kernels
ncu --set basic python matmul_bench.py

# Full metrics on attention kernels only
ncu --set full --kernel-name ".*mm.*|.*attention.*" --launch-count 3 python infer.py

# Roofline metrics (memory-bound vs compute-bound)
ncu --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed \
    --kernel-name ".*mm.*" python matmul_bench.py

# Save report for GUI
ncu -o ncu_report --import-source yes python infer.py
ncu-ui ncu_report.ncu-rep

# List all available metrics
ncu --query-metrics | grep memory
```

**Roofline interpretation:**
- `dram__throughput ≈ 100%` → memory-bound (reduce data movement)
- `sm__throughput ≈ 100%` → compute-bound (need more FLOPs throughput)

---

## PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity

# Basic table (sort by CUDA time)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True) as prof:
    model(x)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

# Chrome trace (open in chrome://tracing)
prof.export_chrome_trace("trace.json")

# TensorBoard (run: tensorboard --logdir tb_log)
with profile(on_trace_ready=tensorboard_trace_handler("tb_log")):
    model(x)

# Quick combined report
python -m torch.utils.bottleneck script.py
```

---

## GPU Monitoring — nvidia-smi

```bash
# Live GPU util + memory (refresh every 500ms)
watch -n 0.5 nvidia-smi

# CSV polling (log to file)
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
           --format=csv -l 1 | tee gpu_log.csv

# Stream memory + utilization every 1 second
nvidia-smi dmon -s mu -d 1

# Rich TUI per-process breakdown
nvitop -m full

# Check NVLink topology
nvidia-smi topo -m
nvidia-smi nvlink --status

# GPU properties
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
```

---

## CPU Flamegraphs — py-spy + FlameGraph

```bash
# Profile a running script → SVG flamegraph
py-spy record -o flamegraph.svg -- python train.py

# Attach to running process
py-spy record -o flamegraph.svg --pid $(pgrep -f train.py) --duration 30

# Generate folded format (for diff flamegraph)
py-spy record -o before.folded --format raw -- python slow_training.py
py-spy record -o after.folded  --format raw -- python fast_training.py

# Differential flamegraph (blue=faster, red=slower)
~/FlameGraph/difffolded.pl before.folded after.folded \
    | ~/FlameGraph/flamegraph.pl > diff.svg
```

---

## Linux perf

```bash
# Hardware counter summary
perf stat -e cycles,instructions,cache-misses,branch-misses python train.py

# CPU profile with call graphs → report
perf record -g -F 99 python train.py
perf report --stdio | head -40

# Generate flamegraph from perf
perf record -g -F 99 python train.py
perf script | ~/FlameGraph/stackcollapse-perf.pl \
           | ~/FlameGraph/flamegraph.pl > cpu_flame.svg

# Live top sorted by cache misses
perf top -e cache-misses --sort comm,dso

# Scheduler latency
perf sched record sleep 5 && perf sched latency
```

---

## eBPF / BCC Tools

```bash
# Count read() syscalls per process
bpftrace -e 'tracepoint:syscalls:sys_enter_read { @[comm] = count(); }'

# Block I/O latency histogram
biolatency -D

# Trace file opens by Python
opensnoop -p $(pgrep python)

# CPU run queue latency (scheduler starvation)
runqlat

# Function call latency distribution
funclatency 'c:read'

# TCP connections from Python
tcpconnect -p $(pgrep python)
```

---

## strace / ltrace

```bash
# Syscall count summary
strace -c python train.py

# Only I/O syscalls on a running process
strace -e trace=read,write,openat -p $(pgrep python)

# Slowest file opens
strace -T -e openat python train.py 2>&1 | sort -t'<' -k2 -n | tail -20

# Library call tracing
ltrace -e malloc -p $(pgrep python) 2>&1 | head -30
```

---

## vmstat / iostat / sar

```bash
# CPU states + memory + swap (1 second intervals)
vmstat 1 30

# Disk I/O throughput and utilization
iostat -xz 1

# CPU utilization history
sar -u 1 60

# Memory utilization history
sar -r 1 60

# Per-partition NVMe stats
iostat -d -p nvme0n1 1

# Save sar data for later analysis
sar -A -o /tmp/sar.data 1 60
sadf /tmp/sar.data
```

**Warning signs:**
- `vmstat wa` > 20% → I/O bottleneck
- `vmstat si/so` > 0 → swap in use (memory pressure)
- `iostat %util` → 100% → disk saturated

---

## NUMA & CPU Affinity

```bash
# Show NUMA topology
numactl --hardware

# Bind process to NUMA node 0
numactl --cpunodebind=0 --membind=0 python train.py

# Restrict DataLoader workers to cores 0-7
taskset -c 0-7 python train.py

# Visualise CPU/cache/NUMA/GPU topology
lstopo --of png > topology.png

# NUMA memory stats for process
numastat -p $(pgrep python)

# Which NUMA node is the GPU on?
cat /sys/bus/pci/devices/0000:01:00.0/numa_node
```

---

## vLLM Commands

```bash
# Start server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --dtype float16 \
    --gpu-memory-utilization 0.85

# With prefix caching
python -m vllm.entrypoints.openai.api_server \
    --model <model> \
    --enable-prefix-caching

# With chunked prefill
python -m vllm.entrypoints.openai.api_server \
    --model <model> \
    --enable-chunked-prefill \
    --max-num-batched-tokens 2048

# Throughput benchmark
python benchmarks/benchmark_throughput.py \
    --backend vllm --model <model> --num-prompts 1000

# Serving benchmark
python benchmarks/benchmark_serving.py \
    --backend vllm --host localhost --port 8000 \
    --dataset sharegpt --request-rate 10
```

---

## Distributed Training / Inference

```bash
# Launch 2-GPU inference
torchrun --nproc_per_node=2 infer_distributed.py

# NCCL all-reduce benchmark
nccl-tests/build/all_reduce_perf -b 8 -e 256M -f 2 -g 2

# Profile NCCL communication
NCCL_DEBUG=INFO nsys profile --trace=cuda,nvtx,nccl \
    torchrun --nproc_per_node=2 train.py

# GPU topology
nvidia-smi topo -m
```

---

## TensorRT

```bash
# Build FP16 engine from ONNX
trtexec --onnx=model.onnx --fp16 --saveEngine=model.trt

# INT8 calibration
trtexec --onnx=model.onnx --int8 --calib=calib_data/ --saveEngine=model_int8.trt

# Benchmark saved engine
trtexec --loadEngine=model.trt --batch=32 --avgRuns=100 --percentile=99

# Validate TRT vs ONNX
polygraphy run model.onnx --trt --fp16 --rtol 1e-3
```

---

## PyTorch Memory

```python
# Check memory
torch.cuda.memory_allocated() / 1e9      # GB actively used
torch.cuda.memory_reserved() / 1e9       # GB allocator holds
torch.cuda.max_memory_allocated() / 1e9  # peak since last reset

# Full detailed summary
print(torch.cuda.memory_summary())

# Reset peak stats before benchmarking
torch.cuda.reset_peak_memory_stats()

# Free unused cached memory
torch.cuda.empty_cache()

# Tune allocator to reduce fragmentation
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
```

---

## Quick Diagnostic Checklist

When something is slow, run in this order:

```bash
# 1. Is the GPU being used at all?
watch -n 0.5 nvidia-smi                          # check util %

# 2. Where is time going? (system view)
nsys profile --stats=true python script.py       # timeline

# 3. Which kernel is slow? (kernel view)
ncu --set basic python script.py                 # per-kernel

# 4. Is CPU starving the GPU?
py-spy record -o flame.svg -- python script.py  # CPU flamegraph

# 5. Is disk I/O the bottleneck?
iostat -xz 1                                     # disk util
vmstat 1                                          # iowait %

# 6. Is memory pressure causing swapping?
vmstat 1 | grep -v 0                             # check si/so columns
```
