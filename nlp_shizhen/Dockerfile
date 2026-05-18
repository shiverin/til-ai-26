# CUDA-enabled PyTorch base image with C++ development tools.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Step 1: Python dependencies (cached unless requirements.txt changes).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# FlashInfer JIT-compiles CUDA kernels at startup that need CUDA 13 bf16
# intrinsics — our base image is CUDA 12.1, so the compile fails (this is what
# crashed submission #5). Uninstall it; vLLM falls back to Triton attention +
# its native sampler, which compile cleanly on CUDA 12.1.
RUN pip uninstall -y flashinfer-python flashinfer 2>/dev/null || true

# Step 2: Pre-downloaded model weights as their own large, stable layer.
# Cached and reused across code changes so editing code does not re-copy ~11 GB.
COPY models ./models

# Step 3: Application code last — small, fast layer that re-runs on edits.
COPY src ./src

EXPOSE 5004

ENV PYTHONPATH=/app
# Reduce CUDA allocator fragmentation.
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Force Triton attention backend (defence-in-depth: even if FlashInfer were
# present, this prevents vLLM from selecting it).
ENV VLLM_ATTENTION_BACKEND=TRITON_ATTN
CMD ["python", "-m", "uvicorn", "src.nlp_server:app", "--host", "0.0.0.0", "--port", "5004"]
