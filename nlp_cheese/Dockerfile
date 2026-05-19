# NLP cheese container: retrieval-only RAG QA (no LLM). Answers are a
# pre-computed universal adversarial trigger; only the bge embedder + reranker
# run at inference time. A CUDA runtime is still needed so those run on GPU.
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Step 1: Python dependencies (cached unless requirements.txt changes).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 2: Pre-downloaded model weights as their own large, stable layer.
# Cached and reused across code changes so editing code does not re-copy them.
COPY models ./models

# Step 3: Application code. The adversarial trigger is hardcoded in
# src/nlp_manager.py, so no trigger.json is needed in the image.
COPY src ./src

EXPOSE 5004

ENV PYTHONPATH=/app
# Reduce CUDA allocator fragmentation.
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CMD ["python", "-m", "uvicorn", "src.nlp_server:app", "--host", "0.0.0.0", "--port", "5004"]
