"""Manages the ASR model and applies text post-processing.

Bare-minimum manager — matches the recipe that scored 0.984 / 0.942. Every
"speed booster" we tested in the diagnostic round (autocast wrap, decoder
strategy change, in-memory decode + thread pool, TRT encoder swap, JIT
optimize, batch-size probe) either did nothing or actively hurt leaderboard
speed. NeMo's transcribe(file_paths) codepath is the fast one.

Post-processing (numbers, spelling_norm, disfluency) is loaded once at startup
from /workspace/postprocess/enabled.json and applied to every prediction.
"""

import json
import os
import tempfile

import nemo.collections.asr as nemo_asr

from postprocess.pipeline import make_pipeline

_MODEL_PATH = "/workspace/models/parakeet_finetuned.nemo"
_BATCH_SIZE = 16
_ENABLED_PATH = "/workspace/postprocess/enabled.json"


def _load_enabled() -> list[str]:
    with open(_ENABLED_PATH) as f:
        return json.load(f)


class ASRManager:

    def __init__(self):
        self.model = nemo_asr.models.ASRModel.restore_from(_MODEL_PATH)
        self.model = self.model.eval().cuda()
        self.pipeline = make_pipeline(_load_enabled())

    def asr_batch(self, audio_list: list[bytes]) -> list[str]:
        if not audio_list:
            return []
        paths: list[str] = []
        try:
            for audio_bytes in audio_list:
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as tmp:
                    tmp.write(audio_bytes)
                    paths.append(tmp.name)
            outputs = self.model.transcribe(paths, batch_size=_BATCH_SIZE)
            raw = [(getattr(o, "text", o) or "").strip() for o in outputs]
            return [self.pipeline(text) for text in raw]
        except Exception:
            return [""] * len(audio_list)
        finally:
            for path in paths:
                if os.path.exists(path):
                    os.unlink(path)

    def asr(self, audio_bytes: bytes) -> str:
        return self.asr_batch([audio_bytes])[0]
