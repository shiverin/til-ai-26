# CUDA-enabled PyTorch base image. Needs CUDA toolkit >= 12.4 because
# flashinfer's JIT compile uses bf16 intrinsics (e.g. __bfloat16_as_ushort)
# that were added after CUDA 12.1. vLLM 0.21 hard-imports flashinfer in its
# sampler, so we cannot skip it — the base must satisfy its CUDA needs.
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Step 1: Python dependencies (cached unless requirements.txt changes).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 2: Pre-downloaded model weights as their own large, stable layer.
# Cached and reused across code changes so editing code does not re-copy ~11 GB.
COPY models ./models

# Step 3: Application code last — small, fast layer that re-runs on edits.
COPY src ./src

EXPOSE 5004

# Put both /app (package import `src.nlp_server`) and /app/src (top-level import
# `nlp_server`) on the path. The finals grader starts the container with
# `uvicorn nlp_server:app` (top-level, like asr_server/noise_server), NOT the
# CMD below, so `nlp_server` must be importable without the `src.` prefix.
ENV PYTHONPATH=/app:/app/src
# Reduce CUDA allocator fragmentation.
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CMD ["python", "-m", "uvicorn", "src.nlp_server:app", "--host", "0.0.0.0", "--port", "5004"]
