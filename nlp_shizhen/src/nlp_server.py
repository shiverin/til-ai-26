"""FastAPI server for the NLP RAG QA container.

Contract (unchanged from the task spec):
  POST /nlp    first call carries {"documents": [...]} -> corpus load
               later calls carry {"question": ...}     -> answers
  GET  /health -> 200 {"message": "health ok"} once ready; 503 while loading

The model loads in a background thread. /health is a real readiness check: it
responds immediately, but reports 503 while the model is still loading and 503
(with the error) if the load failed -- it returns "health ok" only once the
model is genuinely ready to serve.
"""
import asyncio
import logging
import os
import sys
import threading
from typing import Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from nlp_manager import NLPManager

app = FastAPI()
logger = logging.getLogger(__name__)

manager: Optional[NLPManager] = None
model_ready = threading.Event()
model_error: Optional[str] = None


def _init_manager() -> None:
    global manager, model_error
    try:
        manager = NLPManager()
    except Exception as e:
        logger.exception("Model load failed")
        model_error = str(e)
    finally:
        model_ready.set()


threading.Thread(target=_init_manager, daemon=True).start()


class _LoadState:
    def __init__(self) -> None:
        self.status = "idle"  # idle | loading | loaded | failed
        self.task: Optional[asyncio.Task] = None
        self.lock = asyncio.Lock()


load_state = _LoadState()


def _do_load(documents) -> bool:
    model_ready.wait()
    if manager is None:
        raise RuntimeError(f"Model failed to load: {model_error}")
    manager.load_corpus(documents)
    return manager.loaded


async def _load_task(documents) -> None:
    try:
        ok = await asyncio.to_thread(_do_load, documents)
        load_state.status = "loaded" if ok else "failed"
    except Exception:
        logger.exception("Corpus load failed")
        load_state.status = "failed"


@app.post("/nlp")
async def nlp(request: Request) -> dict:
    inputs_json = await request.json()
    first = inputs_json["instances"][0]

    if first.get("documents") is not None:
        async with load_state.lock:
            if load_state.status == "idle":
                load_state.status = "loading"
                load_state.task = asyncio.create_task(
                    _load_task(first["documents"]))
            return {"predictions": [load_state.status]}

    if first.get("poll") is not None:
        return {"predictions": [load_state.status]}

    predictions = [
        await asyncio.to_thread(manager.qa, instance["question"])
        for instance in inputs_json["instances"]
    ]
    return {"predictions": predictions}


@app.get("/health")
def health():
    """Readiness check: 'healthy' only once the model is loaded and ready.

    Returning "health ok" unconditionally would mask a failed model load -- the
    container would look healthy and fail mysteriously later at corpus-load
    time. 503-while-loading is also the standard readiness-probe behaviour that
    `til test` (5-minute poll) and Vertex AI expect.
    """
    if model_error is not None:
        return JSONResponse(
            status_code=503,
            content={"message": f"model load failed: {model_error}"})
    if not model_ready.is_set():
        return JSONResponse(status_code=503, content={"message": "loading"})
    return {"message": "health ok"}
