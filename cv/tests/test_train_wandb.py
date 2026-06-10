"""Verify the W&B integration is properly gated by WANDB_API_KEY."""
from types import SimpleNamespace


def test_maybe_wandb_init_returns_none_without_api_key(monkeypatch):
    """No WANDB_API_KEY => return None (no wandb run, no network call).

    Note: we don't assert `"wandb" not in sys.modules`, because ultralytics
    may transparently import wandb during its own settings init when the
    integration is enabled. The load-bearing invariant is that no wandb
    *run* is started without an explicit API key.

    We set the env var to empty string (rather than unset) so that train.py's
    `load_dotenv(override=False)` autoload cannot re-inject a real key from
    cv/.env — empty counts as "already set" to dotenv, but as "falsy" to the
    gate.
    """
    monkeypatch.setenv("WANDB_API_KEY", "")

    from train import _maybe_wandb_init

    args = SimpleNamespace(
        optimizer="SGD", lr0=0.01, lr0_stage2=0.005,
        epochs_stage1=10, epochs_stage2=30, patience=15, freeze=10,
        imgsz=1280, batch=4, no_weighted=False, noise_aug=False, smoke=False,
    )
    assert _maybe_wandb_init(args) is None


def test_maybe_log_artifact_noop_when_no_run():
    """A None run is silently ignored."""
    from train import _maybe_log_artifact

    # Both branches: no run, and run present but file missing.
    _maybe_log_artifact(None, "/nonexistent/best.pt", 0.42)
    _maybe_log_artifact(object(), "/nonexistent/best.pt", 0.42)
