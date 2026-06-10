"""Manages the ASR model.

BARE-MINIMUM diagnostic version — identical in spirit to the original
asr_manager.py that scored 0.943 speed days ago. All later additions (autocast,
in-memory decode + thread pool, CUDA-graph TDT decoder change, TRT encoder
swap, JIT optimize, batch-size probe, sort-by-length, warmup) have been
removed.

If this submission still lands at ~0.93 speed, the leaderboard floor has
genuinely moved and none of those additions were the culprit. If it lands at
~0.94, one of those additions was costing us — we can layer them back
selectively. The full version is preserved at asr_manager_full.py.
"""

import os
import tempfile

import nemo.collections.asr as nemo_asr

_MODEL_PATH = "/workspace/models/parakeet_finetuned.nemo"
_BATCH_SIZE = 16


class ASRManager:

    def __init__(self):
        self.model = nemo_asr.models.ASRModel.restore_from(_MODEL_PATH)
        self.model = self.model.eval().cuda()

    def asr_batch(self, audio_list: list[bytes]) -> list[str]:
        """Transcribes a list of WAV audio files given as raw bytes."""
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
            return [(getattr(o, "text", o) or "").strip() for o in outputs]
        except Exception:
            return [""] * len(audio_list)
        finally:
            for path in paths:
                if os.path.exists(path):
                    os.unlink(path)

    def asr(self, audio_bytes: bytes) -> str:
        """Transcribes a single WAV audio file given as raw bytes."""
        return self.asr_batch([audio_bytes])[0]
