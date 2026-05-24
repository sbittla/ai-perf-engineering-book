FROM nvcr.io/nvidia/pytorch:25.01-py3

# Ubuntu 24.04 (Noble) package names — bcc-tools was renamed to bpfcc-tools
RUN apt-get update && apt-get install -y \
    linux-tools-generic \
    linux-tools-common \
    bpfcc-tools \
    bpftrace \
    sysstat \
    numactl && \
    rm -rf /var/lib/apt/lists/*

# FlameGraph (Brendan Gregg)
RUN git clone --depth 1 https://github.com/brendangregg/FlameGraph /root/FlameGraph

# Python profilers and GPU monitoring
RUN pip install --no-cache-dir vllm nvitop py-spy

ENV PATH="/root/FlameGraph:$PATH"
