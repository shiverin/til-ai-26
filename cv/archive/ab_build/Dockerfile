# Speed A/B: same code + weights as slim-v8m-fp16, but base image has
# SYSTEM CUDA + cuDNN (not pip wheel CUDA). Tests whether the slim base's
# bundled cu121 libs are the speed culprit vs nvcr's optimized PyTorch.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

# Minimal system deps (OpenCV runtime needs these even in headless mode).
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Base already has torch==2.5.1+cu121, torchvision, numpy, etc.
# Install only what's missing. Use --no-deps for ultralytics to prevent it
# from yanking torch onto a different version.
RUN pip install \
        opencv-python-headless==4.10.0.84 \
        pillow==11.0.0 \
        pyyaml requests scipy tqdm psutil py-cpuinfo \
        matplotlib pandas seaborn ultralytics-thop \
        fastapi==0.115.12 \
        "uvicorn[standard]==0.34.2" \
 && pip install --no-deps ultralytics==8.4.50

COPY cv_manager.py cv_server.py ./
COPY best.pt ./best.pt

EXPOSE 5002
CMD ["uvicorn", "cv_server:app", "--port", "5002", "--host", "0.0.0.0"]
