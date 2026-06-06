"""Hybrid evaluation: a frozen-hybrid opponent/eval agent, the paired-continuation
A/B acceptance harness (the spec's north-star metric), and intervention
diagnostics. Reuses evaluate.evaluate_policy unchanged. train_selfplay.py and
evaluate.py are untouched.
"""
import numpy as np

from features import FeatureBuilder
from hybrid_controller import ActorRuntime, HybridController
from scripted.handover import HandoverTrigger
from scripted.strategies import STRATEGIES


class HybridAgent:
    """A frozen trained hybrid (scripted opener -> RL post-opener) as a self-play
    opponent or eval agent. Runtime-agnostic: pass a torch `ActorRuntime` (samples)
    or an `OnnxActorRuntime` (deterministic argmax — use this for acceptance eval).
    A raw `SymbolicTransformerActor` is wrapped in a torch `ActorRuntime`."""

    def __init__(self, actor, opener=None, trigger=None, post_params=None,
                 name="hybrid"):
        self._runtime = actor if hasattr(actor, "query") else ActorRuntime(actor)
        self._opener = opener
        self._trigger = trigger
        self._post = post_params
        self.name = name
        self.reset()

    def reset(self):
        self.controller = HybridController(
            self._runtime, self._trigger or HandoverTrigger(),
            opener=self._opener, post_params=self._post,
            feature_builder=FeatureBuilder(), forward_bias=0.0)

    def action(self, observation):
        action, _decision = self.controller.step(observation)
        return int(action)


def intervention_rates(buf):
    """Summarize a HybridRolloutBuffer's controller behavior (for diagnostics):
    how often the actor was queried vs forced-escaped, and how often a gate
    overrode the actor's proposal."""
    n = int(buf.size)
    if n == 0:
        return {"n": 0, "actor_query_rate": 0.0, "forced_escape_rate": 0.0,
                "gate_override_rate": 0.0, "proposal_executed_disagreement": 0.0}
    aq = np.asarray(buf.actor_queried, bool)
    disagree = np.asarray(buf.proposed_actions) != np.asarray(buf.executed_actions)
    return {
        "n": n,
        "actor_query_rate": float(aq.mean()),
        "forced_escape_rate": float((~aq).mean()),
        "gate_override_rate": float((aq & disagree).mean()),
        "proposal_executed_disagreement": float(disagree.mean()),
    }


def summarize_paired_deltas(deltas, n_boot=2000, ci=0.95, seed=0):
    """Mean + bootstrap CI over paired (B - A) deltas. CI excluding 0 on the low
    side is the acceptance signal."""
    d = np.asarray(deltas, np.float64)
    if d.size == 0:
        return {"n": 0, "mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(d, size=d.size, replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return {"n": int(d.size), "mean": float(d.mean()),
            "ci_lo": float(lo), "ci_hi": float(hi)}


def paired_continuation_eval(hybrid_agent, seeds,
                             opener_name="balanced_extreme_opening",
                             opponents=None):
    """Paired A/B acceptance (spec north star). Arm A: the scripted opener runs the
    whole episode; arm B: the SAME opener until handover, then `hybrid_agent`. Same
    seeds + opponents, so pre-handover is identical and the per-seed env-return
    delta (B - A) isolates the post-handover difference.

    For deploy-faithful acceptance, build `hybrid_agent` with an OnnxActorRuntime
    (argmax). Returns {deltas, per_seed_a, per_seed_b, mean_a, mean_b, **summary}.
    """
    from evaluate import ScriptedAgent, evaluate_policy   # local: avoid import cycle
    a_agent = ScriptedAgent(opener_name)
    if opponents is None:
        opponents = [ScriptedAgent("balanced") for _ in range(5)]
    res_a = evaluate_policy(a_agent, opponents, seeds)
    res_b = evaluate_policy(hybrid_agent, opponents, seeds)
    deltas = [float(b) - float(a)
              for a, b in zip(res_a.per_seed_scores, res_b.per_seed_scores)]
    out = {"deltas": deltas,
           "per_seed_a": list(res_a.per_seed_scores),
           "per_seed_b": list(res_b.per_seed_scores),
           "mean_a": float(res_a.mean_score),
           "mean_b": float(res_b.mean_score)}
    out.update(summarize_paired_deltas(deltas))
    return out
