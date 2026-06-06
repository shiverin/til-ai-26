import numpy as np
import torch

from train_hybrid import train_hybrid
from hybrid_ppo import HybridPPOConfig
from scripted.handover import HandoverTrigger


def test_train_hybrid_runs_and_updates_weights():
    cfg = HybridPPOConfig(num_minibatches=2, update_epochs=1)
    actor, critic, history = train_hybrid(
        total_updates=2, episodes_per_update=1, learner_slots=("agent_0",),
        seed0=0, cfg=cfg, trigger=HandoverTrigger(step_fallback=5),
        d_model=16, n_layers=1, n_heads=2, device="cpu")
    assert len(history) == 2
    for m in history:
        assert np.isfinite(m["policy_loss"]) and np.isfinite(m["value_loss"])
        assert m["n_active"] <= m["size"]
        assert "forward_bias" in m and "ent_coef" in m


def test_critic_warmup_freezes_actor():
    cfg = HybridPPOConfig(num_minibatches=2, update_epochs=1)
    actor, critic, history = train_hybrid(
        total_updates=1, episodes_per_update=1, learner_slots=("agent_0",),
        seed0=0, cfg=cfg, trigger=HandoverTrigger(step_fallback=5),
        critic_warmup=1, d_model=16, n_layers=1, n_heads=2, device="cpu")
    assert history[0].get("warmup") is True
