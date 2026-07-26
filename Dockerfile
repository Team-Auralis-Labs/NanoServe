# syntax=docker/dockerfile:1

# ---- Builder (CPU / OpenCL) ----
FROM ubuntu:24.04 AS builder-cpu
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    curl build-essential cmake python3 python3-pip python3-venv \
    ocl-icd-opencl-dev opencl-headers \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
WORKDIR /src
COPY allocator allocator/
RUN cd allocator && cargo build --release
COPY engine engine/
RUN cd engine && mkdir -p build && cd build && \
    cmake .. -DNANOSERVE_ENABLE_CUDA=OFF -DNANOSERVE_ENABLE_OPENCL=ON && \
    make -j"$(nproc)"
COPY pyproject.toml README.md ./
COPY nanoserve nanoserve/
RUN python3 -m pip install --no-cache-dir build && \
    python3 -m build --wheel && mkdir -p /wheels && mv dist/*.whl /wheels/

# ---- Builder (CUDA + OpenCL) ----
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder-gpu
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    curl build-essential cmake python3 python3-pip python3-venv \
    ocl-icd-opencl-dev opencl-headers \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
WORKDIR /src
COPY allocator allocator/
RUN cd allocator && cargo build --release
COPY engine engine/
RUN cd engine && mkdir -p build && cd build && \
    cmake .. -DNANOSERVE_ENABLE_CUDA=ON -DNANOSERVE_ENABLE_OPENCL=ON && \
    make -j"$(nproc)"
COPY pyproject.toml README.md ./
COPY nanoserve nanoserve/
RUN python3 -m pip install --no-cache-dir build && \
    python3 -m build --wheel && mkdir -p /wheels && mv dist/*.whl /wheels/

# ---- Runtime CPU ----
FROM python:3.11-slim AS runtime-cpu
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ocl-icd-libopencl1 && \
    rm -rf /var/lib/apt/lists/*
COPY --from=builder-cpu /src/allocator/target/release/libbuddy_alloc.so /opt/nanoserve/lib/
COPY --from=builder-cpu /src/engine/build/libnanoserve_engine.so /opt/nanoserve/lib/
COPY --from=builder-cpu /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && pip install --no-cache-dir "nanoserve[server]"
COPY server server/
COPY tui tui/
COPY examples examples/
ENV LD_LIBRARY_PATH=/opt/nanoserve/lib
ENV NANOSERVE_ENGINE_LIB=/opt/nanoserve/lib/libnanoserve_engine.so
WORKDIR /app/server
EXPOSE 8000
CMD ["python3", "main.py"]

# ---- Runtime GPU ----
FROM nvidia/cuda:12.4.1-base-ubuntu22.04 AS runtime-gpu
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip libgomp1 ocl-icd-libopencl1 && \
    rm -rf /var/lib/apt/lists*
COPY server server/
COPY tui tui/
COPY examples examples/
ENV LD_LIBRARY_PATH=/opt/nanoserve/lib
ENV NANOSERVE_ENGINE_LIB=/opt/nanoserve/lib/libnanoserve_engine.so
WORKDIR /app/server
EXPOSE 8000
CMD ["python3", "main.py"]

# ---- Runtime GGUF (optional llama-cpp-python) ----
FROM runtime-cpu AS runtime-gguf
RUN pip install --no-cache-dir "llama-cpp-python>=0.2.90"
ENV NANOSERVE_DEFAULT_FORMAT=auto
ENV NANOSERVE_GGUF_N_CTX=2048
ENV NANOSERVE_GGUF_N_BATCH=512
