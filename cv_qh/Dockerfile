# RF-DETR submission image for TIL CV.
#
# Mirrors cv/Dockerfile but installs RF-DETR's inference-only stack instead of
# Ultralytics. Bake the chosen checkpoint into src/weights/best.pt before
# building (see promote_checkpoint.sh).

FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /workspace

# RF-DETR/supervision pull cv2 at import; provide its system libs upfront.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip
COPY requirements.docker.txt .
RUN pip install --no-cache-dir -r requirements.docker.txt

# Copies your source files (includes src/weights/best.pt — promote a checkpoint
# into that path with promote_checkpoint.sh before building).
COPY src ./src

CMD ["uvicorn", "src.cv_server:app", "--port", "5002", "--host", "0.0.0.0"]
