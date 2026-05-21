"""Unit tests for ASRManager helpers that don't require a loaded model."""

from unittest.mock import MagicMock

import numpy as np
import torch
from omegaconf import OmegaConf

from asr.src.asr_manager import ASRManager


def _fake_model_with_decoding_cfg() -> MagicMock:
    """Build a model stub exposing the OmegaConf decoding config NeMo uses."""
    cfg = OmegaConf.create(
        {"decoding": {"strategy": "greedy", "greedy": {}}}
    )
    model = MagicMock()
    model.cfg = cfg
    return model


def test_enable_cuda_graph_decoder_sets_flags_and_calls_change():
    """Happy path: helper mutates config and invokes change_decoding_strategy."""
    manager = ASRManager.__new__(ASRManager)  # bypass __init__
    manager.model = _fake_model_with_decoding_cfg()

    applied = manager._try_enable_cuda_graph_decoder()

    assert applied is True
    manager.model.change_decoding_strategy.assert_called_once()
    passed_cfg = manager.model.change_decoding_strategy.call_args.args[0]
    assert passed_cfg.strategy == "greedy_batch"
    assert passed_cfg.greedy.use_cuda_graph_decoder is True


def test_enable_cuda_graph_decoder_swallows_and_logs_failure(caplog):
    """Fallback path: if NeMo rejects the flag, helper returns False, no raise."""
    manager = ASRManager.__new__(ASRManager)
    manager.model = _fake_model_with_decoding_cfg()
    manager.model.change_decoding_strategy.side_effect = RuntimeError(
        "unknown key use_cuda_graph_decoder"
    )

    with caplog.at_level("WARNING", logger="asr.src.asr_manager"):
        applied = manager._try_enable_cuda_graph_decoder()

    assert applied is False
    assert any(
        rec.name == "asr.src.asr_manager"
        and "cuda-graph decoder" in rec.message.lower()
        for rec in caplog.records
    )


def test_asr_batch_returns_predictions_in_input_order(monkeypatch):
    """Length-sort: predictions must come back in the caller's input order."""
    manager = ASRManager.__new__(ASRManager)
    manager._batch_size = 4

    fake_waves = [np.zeros(3, dtype=np.float32),
                  np.zeros(1, dtype=np.float32),
                  np.zeros(2, dtype=np.float32)]
    monkeypatch.setattr(
        ASRManager, "_decode",
        staticmethod(lambda b: fake_waves[int(b)]),
    )

    def fake_transcribe(signals, batch_size):
        return [type("H", (), {"text": f"len={len(s)}"})() for s in signals]

    manager.model = type("M", (), {"transcribe": staticmethod(fake_transcribe)})

    inputs = [b"0", b"1", b"2"]
    out = manager.asr_batch(inputs)

    assert out == ["len=3", "len=1", "len=2"]


def test_asr_batch_sorts_signals_before_transcribing(monkeypatch):
    """Length-sort: transcribe() must see signals ascending by length."""
    manager = ASRManager.__new__(ASRManager)
    manager._batch_size = 4

    fake_waves = [np.zeros(5, dtype=np.float32),
                  np.zeros(1, dtype=np.float32),
                  np.zeros(3, dtype=np.float32)]
    monkeypatch.setattr(
        ASRManager, "_decode",
        staticmethod(lambda b: fake_waves[int(b)]),
    )

    seen_lengths = []
    def fake_transcribe(signals, batch_size):
        seen_lengths.extend(len(s) for s in signals)
        return [type("H", (), {"text": "x"})() for _ in signals]

    manager.model = type("M", (), {"transcribe": staticmethod(fake_transcribe)})
    manager.asr_batch([b"0", b"1", b"2"])

    assert seen_lengths == [1, 3, 5]


def test_asr_batch_empty_input_returns_empty():
    """Edge case: empty list does not crash and returns []."""
    manager = ASRManager.__new__(ASRManager)
    manager._batch_size = 4
    manager.model = type("M", (), {"transcribe": staticmethod(lambda *a, **k: [])})

    assert manager.asr_batch([]) == []
