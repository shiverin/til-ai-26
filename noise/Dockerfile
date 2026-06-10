# Dockerfile for the adversarial-noise image.
FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /workspace

# cv2 is shipped in NGC base as opencv-python (not headless); install
# system deps so it imports.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY weights ./weights

# The finals grader runs `uvicorn noise_server:app` (top-level), so
# noise_server must be importable without the `src.` prefix.
ENV PYTHONPATH=/workspace:/workspace/src

CMD ["uvicorn", "src.noise_server:app", "--port", "5003", "--host", "0.0.0.0"]
