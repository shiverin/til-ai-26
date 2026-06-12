# Dockerfile for building the CV image.


# The base image, an example deep learning VM.
# For a full list, see https://us-docker.pkg.dev/deeplearning-platform-release/gcr.io/
# For info, see https://cloud.google.com/deep-learning-vm/docs/images#supported-frameworks
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV CV_CONF=0.40
ENV CV_IMGSZ=1536
ENV CV_IOU=0.45
ENV CV_TTA=0
ENV CV_DENOISE=none
ENV CV_SAHI=0
WORKDIR /workspace

# cv2 needs libglib2.0 at import time. libgl1 is kept for any non-headless
# opencv fallback. Install before pip so ultralytics can import cv2.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Installs your dependencies.
RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copies your source files.
COPY src ./src

# Put both /workspace (for `from src.cv_manager import ...`) and /workspace/src
# (top-level import `cv_server`) on the path. The finals grader starts the
# container with `uvicorn cv_server:app` (top-level, like asr_server/noise_server),
# NOT the CMD below, so `cv_server` must be importable without the `src.` prefix.
ENV PYTHONPATH=/workspace:/workspace/src

# Starts your model server.
CMD ["uvicorn", "src.cv_server:app", "--port", "5002", "--host", "0.0.0.0"]