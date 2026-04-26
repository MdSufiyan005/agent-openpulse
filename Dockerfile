# --- Stage 1: Builder ---
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git cmake build-essential python3.10 python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Build llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp.git --depth=1 \
    && cd llama.cpp \
    && cmake -B build -DLLAMA_CURL=OFF -DLLAMA_CUDA=OFF \
    && cmake --build build -j$(nproc) --config Release --target llama-quantize

# Install Python dependencies in builder stage where gcc is available
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip

# 1. Install heavy core dependencies first (cached separately)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --user torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Install remaining requirements
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --user -r requirements.txt

# 3. Install dashboard and optimization tools
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --user modelpulse nni streamlit plotly pandas

# --- Stage 2: Runtime ---
FROM python:3.10-slim

# Install runtime dependencies (libgomp1 is needed for llama.cpp and torch)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy llama.cpp binaries from builder
COPY --from=builder /build/llama.cpp/build/bin/llama-quantize /usr/local/bin/
# Copy llama.cpp shared libraries required by llama-quantize
COPY --from=builder /build/llama.cpp/build/bin/*.so* /usr/local/lib/
# Copy llama.cpp conversion scripts used by setup.py
COPY --from=builder /build/llama.cpp /app/llama.cpp

# Install Python dependencies in runtime image so CLI entrypoints
# (streamlit/modelpulse) are generated against the runtime interpreter.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir nni streamlit plotly pandas

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p artifacts/results artifacts/models-storage artifacts/shards images memory reference_docs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Expose ports
EXPOSE 8000
EXPOSE 8501

CMD ["python", "run.py"]
