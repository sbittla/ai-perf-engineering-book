# Phase 5 — Workload Porting & Benchmarking

**Duration: 4 weeks — directly aligned with JD requirements**

## Module 10: Workload Porting
```bash
# The core porting skill: observe bottleneck shift
python module10_porting/workload_port.py --mode all
python module10_porting/dataloader_worker.py --label default
numactl --cpunodebind=0 python module10_porting/dataloader_worker.py --label numa
```

## Module 11: Benchmarking & Characterisation
```bash
python module11_benchmarking/workload_characterize.py
python module11_benchmarking/throughput_latency_curve.py --plot
```
