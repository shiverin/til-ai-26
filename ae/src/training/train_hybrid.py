"""Hybrid post-opener PPO training loop.

Ties together collect_hybrid_rollout (5.3), the reused GAE + centralized-critic
values, and ppo_update_hybrid (5.2/5.1) with the forward-bias + entropy schedules
and an optional critic warm-up. Builds the actor + critic with dropout=0 so the
ppo_update_hybrid determinism assert holds. train_selfplay.py is untouched.

Opponents here are scripted (opponent_names). The self-play league + paired-
continuation acceptance eval are Plan 5.5.
"""
from dataclasses import replace

import numpy as np
import torch
import torch.nn as nn

from critic import CentralizedCritic
from hybrid_ppo import HybridPPOConfig, forward_bias_value, ppo_update_hybrid
from hybrid_rollout import collect_hybrid_rollout
from policy import SymbolicTransformerActor
from train_selfplay import compute_advantages, critic_values, make_optimizer

GAMMA = 0.99
GAE_LAMBDA = 0.95


def critic_only_update(critic, opt, buf, returns, cfg, device):
    """Warm-up step: fit the centralized critic to `returns`; the actor is left
    untouched (no actor term in the loss -> set_to_none keeps actor grads None ->
    Adam skips the actor group). Mirrors ppo_update_hybrid's value-loss path."""
    gstate = torch.from_numpy(buf.gstate).to(device)
    gscalar = torch.from_numpy(buf.gscalar).to(device)
    ret = torch.from_numpy(np.asarray(returns, np.float32)).to(device)
    n = buf.size
    mb_size = max(1, n // cfg.num_minibatches)
    inds = np.arange(n)
    critic.train()
    vl = torch.zeros((), device=device)
    for _ in range(cfg.update_epochs):
        np.random.shuffle(inds)
        for start in range(0, n, mb_size):
            mb = torch.from_numpy(inds[start:start + mb_size]).to(device)
            v = critic(gstate[mb], gscalar[mb])
            vl = 0.5 * ((v - ret[mb]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            (cfg.vf_coef * vl).backward()
            nn.utils.clip_grad_norm_(
                [p for p in critic.parameters() if p.grad is not None],
                cfg.max_grad_norm)
            opt.step()
    return {"value_loss": float(vl.detach()), "warmup": True}


def train_hybrid(total_updates=3000, episodes_per_update=4,
                 learner_slots=("agent_0",), seed0=0, cfg=None,
                 forward_bias_init=0.0, anti_idle_penalty=0.0, critic_warmup=0,
                 trigger=None, opponent_names=None, post_params=None,
                 d_model=64, n_layers=4, n_heads=4, device="cpu",
                 checkpoint_dir=None, checkpoint_every=0, actor=None, critic=None,
                 **_ignored):
    """Run `total_updates` of hybrid PPO. Returns (actor, critic, history).

    Builds actor + critic with dropout=0 (the ppo_update_hybrid determinism assert
    requires it). `forward_bias` follows the anneal+zero-hold schedule; `ent_coef`
    anneals linearly from cfg.ent_coef to cfg.ent_coef_final. The first
    `critic_warmup` updates fit the critic only (actor frozen). Unknown kwargs are
    ignored (`**_ignored`) so callers can pass extra diagnostics flags."""
    cfg = cfg or HybridPPOConfig()
    actor = actor or SymbolicTransformerActor(
        d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=0.0)
    critic = critic or CentralizedCritic(
        d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=0.0)
    actor.to(device)
    critic.to(device)
    opt = make_optimizer(actor, critic, cfg)
    history = []
    for update in range(total_updates):
        fb = forward_bias_value(update, total_updates, forward_bias_init)
        frac = update / max(1, total_updates - 1)
        ent = cfg.ent_coef + (cfg.ent_coef_final - cfg.ent_coef) * frac
        cfg_u = replace(cfg, ent_coef=ent)
        buf = collect_hybrid_rollout(
            actor, learner_slots, episodes_per_update,
            seed0 + update * episodes_per_update, trigger=trigger,
            post_params=post_params, forward_bias=fb,
            anti_idle_penalty=anti_idle_penalty, opponent_names=opponent_names)
        if buf.size == 0:
            history.append({"update": update, "size": 0, "skipped": True})
            continue
        vals = critic_values(critic, buf.gstate, buf.gscalar, device)
        adv, ret = compute_advantages(buf.rewards, vals, buf.dones,
                                      GAMMA, GAE_LAMBDA)
        if update < critic_warmup:
            m = critic_only_update(critic, opt, buf, ret, cfg_u, device)
            m.setdefault("policy_loss", 0.0)
            m.setdefault("approx_kl", 0.0)
            m["n_active"] = 0
        else:
            m = ppo_update_hybrid(actor, critic, opt, buf, adv, ret, cfg_u, device)
        m["update"] = update
        m["size"] = int(buf.size)
        m["forward_bias"] = fb
        m["ent_coef"] = ent
        m["post_handover_return"] = float(buf.env_rewards.sum()
                                          / max(1, float(buf.dones.sum())))
        history.append(m)
        if (checkpoint_dir and checkpoint_every
                and (update + 1) % checkpoint_every == 0):
            actor.save_checkpoint(f"{checkpoint_dir}/actor_{update + 1}.pt")
    return actor, critic, history
