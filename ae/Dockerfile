# Dockerfile for building the AE image.


# The base image. Using a regular Python base because AE agents don't typically
# require GPU. This will speed up your evaluation.
FROM python:3.11-slim

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /workspace

# Installs your dependencies.
RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copies your source files. `.dockerignore` excludes src/training/ (the
# training pipeline) and other dev-only paths from the image — only the
# runtime modules (scripted/, ae_*, policy.py, features.py, *.json, *.pt,
# *.onnx) end up in /workspace.
COPY src .

# Which agent the server runs. Default "scripted" (proven rule cascade);
# "neural" loads policy.onnx via NeuralAEManager.
#   scripted — rule cascade (see AE_STRATEGY below for which one).
#   neural   — trained BC clone served via ONNX (ignores AE_STRATEGY).
# Override per build: docker build --build-arg AE_MODE=neural ...
# Or per run:         docker run -e AE_MODE=neural ...
#ARG AE_MODE=neural
ARG AE_MODE=scripted
ENV AE_MODE=${AE_MODE}

# Which scripted strategy this image serves when AE_MODE=scripted. Override
# per build: docker build --build-arg AE_STRATEGY=<name> ...
# Must be a key in ae/src/scripted/strategies.py STRATEGIES — one of:
#   balanced, balanced_extreme, base_rusher, base_rusher_extreme,
#   collector, camper, forager, lean_rush, defender, adaptive.
# Or the standalone "greedy" agent (ae/src/scripted/greedy.py).
# Ignored when AE_MODE=neural.
ARG AE_STRATEGY=balanced_extreme_opening
ENV AE_STRATEGY=${AE_STRATEGY}

# Starts your model server.
CMD uvicorn ae_server:app --port 5005 --host 0.0.0.0
#CMD ["python", "-m", "uvicorn", "src.ae_server:app", "--host", "0.0.0.0", "--port", "5005"]
