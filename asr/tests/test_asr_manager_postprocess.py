"""Tests that ASRManager applies the postprocess pipeline to predictions.

The real NeMo model load is expensive and requires a GPU + the 2.5GB .nemo
file. We mock it so the test runs everywhere in a few ms.
"""

import json
import os
import sys
import tempfile
import types
from unittest import mock

# Make `src` and `postprocess` importable as top-level packages.
ASR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ASR_DIR)
sys.path.insert(0, os.path.join(ASR_DIR, "src"))


def _make_fake_nemo(monkeypatch, return_texts):
    """Replaces nemo.collections.asr.models.ASRModel.restore_from."""
    class FakeOutput:
        def __init__(self, text):
            self.text = text

    class FakeModel:
        def eval(self):
            return self
        def cuda(self):
            return self
        def transcribe(self, paths, batch_size=16):
            return [FakeOutput(t) for t in return_texts]

    fake_module = types.SimpleNamespace(
        models=types.SimpleNamespace(
            ASRModel=types.SimpleNamespace(
                restore_from=lambda path: FakeModel()
            )
        )
    )
    # Populate the full nemo module chain in sys.modules so that
    # `import nemo.collections.asr as nemo_asr` inside asr_manager resolves
    # to our fake without trying to import the real (absent) package.
    fake_nemo = types.ModuleType("nemo")
    fake_collections = types.ModuleType("nemo.collections")
    fake_nemo.collections = fake_collections
    fake_collections.asr = fake_module
    monkeypatch.setitem(sys.modules, "nemo", fake_nemo)
    monkeypatch.setitem(sys.modules, "nemo.collections", fake_collections)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr", fake_module)


def _enabled_path_override(monkeypatch, enabled):
    """Point asr_manager._ENABLED_PATH at a tmp file with the given list."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False)
    json.dump(enabled, tmp)
    tmp.close()
    import asr_manager
    monkeypatch.setattr(asr_manager, "_ENABLED_PATH", tmp.name)


def _wav_bytes() -> bytes:
    # A tiny WAV header — payload is irrelevant because NeMo is mocked.
    return b"RIFF\x24\x00\x00\x00WAVE"


def test_predictions_pass_through_pipeline(monkeypatch):
    _make_fake_nemo(monkeypatch, return_texts=[
        "near the harbour gate",
        "the the centre 3",
    ])
    # Force fresh import after mocking, then patch _ENABLED_PATH on that module.
    sys.modules.pop("asr_manager", None)
    import asr_manager
    _enabled_path_override(
        monkeypatch, ["numbers", "spelling_norm", "disfluency"])

    mgr = asr_manager.ASRManager()
    out = mgr.asr_batch([_wav_bytes(), _wav_bytes()])
    assert out == [
        "near the harbor gate",
        "the center three",
    ]


def test_empty_audio_list_returns_empty(monkeypatch):
    _make_fake_nemo(monkeypatch, return_texts=[])
    sys.modules.pop("asr_manager", None)
    import asr_manager
    _enabled_path_override(monkeypatch, [])

    mgr = asr_manager.ASRManager()
    assert mgr.asr_batch([]) == []


def test_pipeline_disabled_passes_text_through_unchanged(monkeypatch):
    _make_fake_nemo(monkeypatch, return_texts=["the the centre 3"])
    sys.modules.pop("asr_manager", None)
    import asr_manager
    _enabled_path_override(monkeypatch, [])

    mgr = asr_manager.ASRManager()
    assert mgr.asr_batch([_wav_bytes()]) == ["the the centre 3"]
