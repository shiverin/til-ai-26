"""Runs the CV server. POST /cv on port 5002, batched inference."""
import base64
from typing import Any

from fastapi import FastAPI, Request

from src.cv_manager import CVManager

app = FastAPI()
# Heads-up: this triggers the in-container TensorRT engine build at module
# import time — fine for the production server, slow if imported elsewhere.
manager = CVManager.from_weights()


@app.post("/cv")
async def cv(request: Request) -> dict[str, list[list[dict[str, Any]]]]:
    inputs_json = await request.json()
    images = [base64.b64decode(inst["b64"]) for inst in inputs_json["instances"]]
    predictions = manager.cv_batch(images)
    return {"predictions": predictions}


@app.get("/health")
def health() -> dict[str, str]:
    return {"message": "health ok"}
