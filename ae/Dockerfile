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

# Copies the submitted scripted runtime.
COPY src .

# Which scripted strategy this image serves. Override per build with
# `--build-arg AE_STRATEGY=<name>`.
# Must be a key in ae/src/scripted/strategies.py STRATEGIES — one of:
#   balanced, balanced_extreme, base_rusher, base_rusher_extreme,
#   collector, camper, forager, lean_rush, defender, adaptive.
# Or the standalone "greedy" agent (ae/src/scripted/greedy.py).
ARG AE_STRATEGY=balanced_extreme_opening
ENV AE_STRATEGY=${AE_STRATEGY}

# Starts your model server.
CMD uvicorn ae_server:app --port 5005 --host 0.0.0.0
