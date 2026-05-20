"""Unit tests for ASRManager helpers that don't require a loaded model."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
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

    with caplog.at_level("WARNING"):
        applied = manager._try_enable_cuda_graph_decoder()

    assert applied is False
    assert any(
        "cuda-graph decoder" in rec.message.lower() for rec in caplog.records
    )
