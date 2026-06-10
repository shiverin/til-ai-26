FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore

WORKDIR /workspace

RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python && \
    pip install --no-cache-dir opencv-python-headless

COPY src .

CMD uvicorn cv_server:app --port 5002 --host 0.0.0.0
