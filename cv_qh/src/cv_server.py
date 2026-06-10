"""RF-DETR CV server. POST /cv on port 5002, batched inference."""
import base64
from typing import Any

from fastapi import FastAPI, Request

from src.cv_manager import CVManager

app = FastAPI()
# Loading + a single warmup inference happens at import time so the first /cv
# request doesn't pay model-init latency.
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
