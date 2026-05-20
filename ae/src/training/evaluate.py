"""Standalone evaluation harness for AE agents.

Runs the novice AEC env over a fixed seed set, scoring a policy-under-test in
slot agent_0 against a configurable opponent set in agent_1..agent_5. Reports
mean score and win-rate. Used to gate the BC stage (spec C §6.1) and rung
promotion (§6.2). Regression checks must always pass the SAME seeds + opponents
to two checkpoints — never compare across opponent sets.
"""
import random
from dataclasses import dataclass

import numpy as np
import torch

from features import FeatureBuilder
from policy import SymbolicTransformerActor
from scripted.belief import Belief
from scripted.decide import act
from scripted.map_prior import MapPrior
from scripted.strategies import STRATEGIES
from til_environment import bomberman_env
from til_environment.config import default_config

EPISODE_LEN = 200


# ----- agent adapters: each maps observation -> action int ---------------- #
class RandomAgent:
    """Uniform-random over legal actions."""
    name = "random"

    def reset(self):
        pass

    def action(self, observation):
        mask = np.asarray(observation["action_mask"], dtype=bool).reshape(-1)
        legal = np.flatnonzero(mask)
        return int(random.choice(legal)) if len(legal) else 4


class ScriptedAgent:
    """A named scripted strategy."""

    def __init__(self, strategy_name="balanced"):
        self.name = f"scripted:{strategy_name}"
        self.strategy = STRATEGIES[strategy_name]
        self.prior = MapPrior.load()
        self.belief = Belief()
        self._started = False

    def reset(self):
        self._started = False

    def action(self, observation):
        step = int(np.asarray(observation["step"]).flat[0])
        if step == 0 or not self._started:
            self.prior.identify_team(observation["base_location"])
            self.belief.reset(self.prior)
            self._started = True
        self.belief.update(observation)
        return int(act(self.belief, observation["action_mask"], self.strategy))


class NeuralAgent:
    """A trained SymbolicTransformerActor served with greedy argmax."""

    def __init__(self, actor, name="neural"):
        self.name = name
        self.actor = actor
        self.actor.eval()
        self.fb = FeatureBuilder()

    def reset(self):
        self.fb = FeatureBuilder()

    def action(self, observation):
        grid, base_feats, raw_agent, raw_base, scalar = self.fb.build(
            observation)
        mask = np.asarray(observation["action_mask"], dtype=bool).reshape(-1)
        with torch.no_grad():
            logits = self.actor(
                torch.from_numpy(grid).unsqueeze(0),
                torch.from_numpy(base_feats).unsqueeze(0),
                torch.from_numpy(raw_agent).unsqueeze(0),
                torch.from_numpy(raw_base).unsqueeze(0),
                torch.from_numpy(scalar).unsqueeze(0),
            )[0].numpy()
        logits = np.where(mask, logits, -1e8)
        return int(np.argmax(logits))


@dataclass
class EvalResult:
    episodes: int
    mean_score: float
    win_rate: float
    per_seed_scores: list


def evaluate_policy(agent, opponents, seeds):
    """Run one episode per seed; `agent` in agent_0, `opponents` in agent_1..5.

    Returns an EvalResult for `agent`. The eval env always uses the UNMODIFIED
    default reward config (spec C §6 reward shaping is training-only).
    """
    cfg = default_config()
    cfg.env.novice = True
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    slots = ["agent_0", "agent_1", "agent_2", "agent_3", "agent_4", "agent_5"]
    by_slot = {slots[0]: agent}
    for i, opp in enumerate(opponents):
        by_slot[slots[i + 1]] = opp

    scores, wins = [], 0
    for seed in seeds:
        random.seed(seed)
        env.reset(seed=seed)
        for a in by_slot.values():
            a.reset()
        totals = {s: 0.0 for s in slots}
        for slot in env.agent_iter():
            obs, reward, term, trunc, _ = env.last()
            totals[slot] += float(reward)
            if term or trunc:
                env.step(None)
                continue
            env.step(by_slot[slot].action(obs))
        my = totals[slots[0]]
        scores.append(my)
        if my > max(totals[s] for s in slots[1:]):
            wins += 1
    env.close()
    return EvalResult(
        episodes=len(seeds),
        mean_score=float(np.mean(scores)) if scores else 0.0,
        win_rate=wins / len(seeds) if seeds else 0.0,
        per_seed_scores=scores,
    )
