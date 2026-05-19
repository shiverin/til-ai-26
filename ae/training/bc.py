"""Stage 1 — behavior cloning with DAgger-style dataset aggregation.

Generate (feature, action) pairs: roll out a mix of a teacher scripted strategy
and an in-training actor (beta = P[teacher acts]); EVERY visited state is
labeled with the teacher strategy's action regardless of who acted. This widens
state coverage beyond the teacher's narrow near-deterministic trajectory.

Slots and opponents are varied across episodes so the parameter-shared actor
sees all 6 positions.
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

SLOTS = ["agent_0", "agent_1", "agent_2", "agent_3", "agent_4", "agent_5"]


@dataclass
class BCSample:
    grid: np.ndarray        # float32 [17,16,16]
    base_feats: np.ndarray  # float32 [5,11]
    raw_agent: np.ndarray   # float32 [7,5,25]
    raw_base: np.ndarray    # float32 [7,7,25]
    scalar: np.ndarray      # float32 [10]
    mask: np.ndarray        # bool    [6]
    action: int             # teacher label


def _teacher_action(belief_cache, slot, observation, strategy):
    """Run the teacher strategy for one slot; belief_cache holds per-slot state."""
    if slot not in belief_cache:
        belief_cache[slot] = (MapPrior.load(), Belief(), [False])
    prior, belief, started = belief_cache[slot]
    step = int(np.asarray(observation["step"]).flat[0])
    if step == 0 or not started[0]:
        prior.identify_team(observation["base_location"])
        belief.reset(prior)
        started[0] = True
    belief.update(observation)
    return int(act(belief, observation["action_mask"], strategy))


def _default_opponent_pool():
    """Opponent-agent factories for BC dataset rollouts (spec C §6.1).

    Random is over-represented so no episode degenerates into an all-identical
    matchup; the scripted strategies add state coverage. Each entry is a
    zero-arg callable returning a fresh agent with .reset()/.action(obs).
    """
    pool = [RandomAgent, RandomAgent, RandomAgent, RandomAgent]
    for _name in ("base_rusher", "base_rusher_extreme", "balanced_extreme",
                  "collector", "camper"):
        pool.append(lambda n=_name: ScriptedAgent(n))
    return pool


def collect_dagger_dataset(teacher_strategy, rollout_policy, beta,
                           num_episodes, seeds, opponent_pool=None):
    """Collect a BC dataset.

    The non-learner slots are driven by a VARIED opponent set sampled per
    episode (spec C §6.1 — "vary the opponents' behavior"). Driving them with
    the teacher instead makes a competent strategy behave degenerately: six
    identical agents in mutual contention oscillate in place and never reach a
    base to bomb, starving the dataset of PLACE_BOMB labels and producing a
    clone structurally unable to bomb.

    Args:
        teacher_strategy: STRATEGIES key — the labeling teacher.
        rollout_policy: a SymbolicTransformerActor used (with prob 1-beta) to drive the
            learner slot, or None to always use the teacher.
        beta: P[teacher drives the learner slot] in [0,1].
        num_episodes: episodes to roll out.
        seeds: per-episode seeds (len == num_episodes).
        opponent_pool: list of zero-arg agent factories for the non-learner
            slots; defaults to `_default_opponent_pool()`.
    Returns:
        list[BCSample].
    """
    if opponent_pool is None:
        opponent_pool = _default_opponent_pool()
    cfg = default_config()
    cfg.env.novice = True
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    strategy = STRATEGIES[teacher_strategy]
    dataset = []
    if rollout_policy is not None:
        rollout_policy.eval()

    for ep in range(num_episodes):
        seed = seeds[ep % len(seeds)]
        random.seed(seed)
        env.reset(seed=seed)
        # rotate which slot the learner occupies for coverage
        learner_slot = SLOTS[ep % len(SLOTS)]
        belief_cache = {}
        fb = FeatureBuilder(teacher_strategy=teacher_strategy)
        # one fresh, independently-sampled opponent per non-learner slot
        opponents = {}
        for s in SLOTS:
            if s != learner_slot:
                opp = random.choice(opponent_pool)()
                opp.reset()
                opponents[s] = opp

        for slot in env.agent_iter():
            obs, reward, term, trunc, _ = env.last()
            if term or trunc:
                env.step(None)
                continue
            if slot == learner_slot:
                teacher_a = _teacher_action(belief_cache, slot, obs, strategy)
                grid, base_feats, raw_agent, raw_base, scalar = fb.build(obs)
                mask = np.asarray(obs["action_mask"], dtype=bool).reshape(-1)
                dataset.append(BCSample(
                    grid=grid, base_feats=base_feats, raw_agent=raw_agent,
                    raw_base=raw_base, scalar=scalar, mask=mask.copy(),
                    action=teacher_a))
                if rollout_policy is not None and random.random() > beta:
                    with torch.no_grad():
                        logits = rollout_policy(
                            torch.from_numpy(grid).unsqueeze(0),
                            torch.from_numpy(base_feats).unsqueeze(0),
                            torch.from_numpy(raw_agent).unsqueeze(0),
                            torch.from_numpy(raw_base).unsqueeze(0),
                            torch.from_numpy(scalar).unsqueeze(0),
                        )[0].numpy()
                    logits = np.where(mask, logits, -1e8)
                    env.step(int(np.argmax(logits)))
                else:
                    env.step(teacher_a)
            else:
                env.step(opponents[slot].action(obs))
    env.close()
    return dataset


import torch.nn as nn
import torch.optim as optim

from evaluate import evaluate_policy, NeuralAgent, ScriptedAgent, RandomAgent


def train_bc(actor, dataset, epochs=20, batch_size=256, lr=1e-3, verbose=False):
    """Cross-entropy fit of `actor` to the teacher labels. Returns loss history.

    Illegal actions are masked before the softmax so the clone never learns to
    favour a move the env forbids. Training runs on whatever device the actor's
    parameters live on — move the actor with `.to(device)` before calling to
    train on GPU. Set `verbose=True` to show a tqdm progress bar with the
    running batch loss and the per-epoch average loss.
    """
    from tqdm.auto import tqdm
    device = next(actor.parameters()).device

    def _stack(field):
        return torch.from_numpy(
            np.stack([getattr(s, field) for s in dataset])).to(device)

    grids = _stack("grid")
    base_feats = _stack("base_feats")
    raw_agents = _stack("raw_agent")
    raw_bases = _stack("raw_base")
    scalars = _stack("scalar")
    masks = _stack("mask")
    labels = torch.tensor([s.action for s in dataset],
                          dtype=torch.long).to(device)
    n = len(dataset)
    opt = optim.Adam(actor.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    actor.train()
    n_batches = (n + batch_size - 1) // batch_size
    pbar = tqdm(total=epochs * n_batches, desc="train_bc",
                disable=not verbose)
    for ep in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            mb = perm[start:start + batch_size]
            logits = actor(grids[mb], base_feats[mb], raw_agents[mb],
                           raw_bases[mb], scalars[mb])
            logits = torch.where(masks[mb], logits,
                                 torch.full_like(logits, -1e8))
            loss = loss_fn(logits, labels[mb])
            opt.zero_grad()
            loss.backward()
            opt.step()
            batch_loss = loss.item()
            epoch_loss += batch_loss * len(mb)
            pbar.update(1)
            pbar.set_postfix(epoch=f"{ep + 1}/{epochs}",
                             loss=f"{batch_loss:.4f}")
        avg = epoch_loss / n
        history.append(avg)
        if verbose:
            tqdm.write(f"  epoch {ep + 1:>2}/{epochs}  avg_loss {avg:.4f}")
    pbar.close()
    return history


def bc_gate(actor, teacher_strategy="balanced", seeds=None, tolerance=0.05):
    """Spec C §6.1 gate: the clone's score must be within `tolerance` of the
    teacher's, BOTH evaluated under identical opponents + seeds (random foes).

    Returns (passed: bool, detail: dict).
    """
    if seeds is None:
        seeds = list(range(16))
    opponents = [RandomAgent() for _ in range(5)]
    clone = evaluate_policy(NeuralAgent(actor, "bc_clone"), opponents, seeds)
    teacher = evaluate_policy(ScriptedAgent(teacher_strategy), opponents, seeds)
    # relative gate — tolerate the clone scoring slightly under the teacher.
    ref = abs(teacher.mean_score) + 1e-9
    rel_gap = (teacher.mean_score - clone.mean_score) / ref
    passed = rel_gap <= tolerance
    return passed, {"clone": clone.mean_score, "teacher": teacher.mean_score,
                    "rel_gap": rel_gap, "tolerance": tolerance}


def main():
    """CLI entry: collect DAgger data, train, gate, save policy_bc.pt."""
    import os
    teacher = os.environ.get("BC_TEACHER", "balanced")  # KNOB
    actor = SymbolicTransformerActor()
    # round 1: pure teacher (beta=1)
    ds = collect_dagger_dataset(teacher, None, 1.0, 24, list(range(24)))
    train_bc(actor, ds, epochs=20)
    # round 2-3: DAgger aggregation with the partially-trained actor
    for rnd in range(2):
        beta = 0.5                                          # KNOB
        more = collect_dagger_dataset(teacher, actor, beta, 24,
                                      list(range(100 + rnd * 24, 124 + rnd * 24)))
        ds += more
        train_bc(actor, ds, epochs=20)
    passed, detail = bc_gate(actor, teacher)
    print(f"BC gate {'PASS' if passed else 'FAIL'}: {detail}")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "src", "policy_bc.pt")
    actor.save_checkpoint(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
