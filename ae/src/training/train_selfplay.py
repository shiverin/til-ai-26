"""Stage 2 — PPO self-play ladder for the AE finals agent.

A CleanRL-style single-file PPO: a parameter-shared
SymbolicTransformerActor controls a configurable subset of the 6 AEC slots; league
opponents fill the rest. Transitions are collected from EVERY learner slot each
step (the ~3-6x data gain). Advantages use a centralized critic over a global
state encoded from the entity registry (see critic.encode_global_state).
Reward shaping (anti-idle, annealed) is training-only.
"""
import copy
import os
import random
import sys
import multiprocessing
from dataclasses import dataclass, field

import numpy as np
import torch

# ae/src holds scripted/, policy.py; conftest.py adds it for tests, a
# standalone run needs this too.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from critic import (CentralizedCritic, encode_global_state, STATE_PLANES,
                    STATE_SCALARS)
from evaluate import RandomAgent, ScriptedAgent, NeuralAgent
from features import (FeatureBuilder, GRID_CHANNELS, NUM_BASES, BASE_FIELDS,
                       RAW_AGENT_SHAPE, RAW_BASE_SHAPE, FEATURE_SCALARS)
from league import League
from policy import SymbolicTransformerActor, NUM_ACTIONS
from scripted.belief import Belief
from scripted.decide import act
from scripted.map_prior import MapPrior
from scripted.strategies import STRATEGIES
from til_environment import bomberman_env
from til_environment.config import default_config

SLOTS = ["agent_0", "agent_1", "agent_2", "agent_3", "agent_4", "agent_5"]
EPISODE_LEN = 200
GRID_SIZE = 16


# ----- opponent registry -------------------------------------------------- #
class OpponentRegistry:
    """Builds per-slot opponent agents from a league Member or a fixed spec.

    A league checkpoint Member loads a frozen SymbolicTransformerActor; a scripted Member
    loads its strategy. `make(member)` returns a stateful agent with reset() and
    action(observation).
    """

    def __init__(self, league):
        self.league = league
        self._actor_cache = {}

    def make(self, member):
        if member is None or member == "random":
            return RandomAgent()
        if member.kind == "scripted":
            return ScriptedAgent(member.ref)
        # checkpoint
        if member.ref not in self._actor_cache:
            actor = SymbolicTransformerActor.from_checkpoint(member.ref)
            actor.eval()
            self._actor_cache[member.ref] = actor
        return NeuralAgent(self._actor_cache[member.ref], name=member.name)


# ----- rollout buffer ----------------------------------------------------- #
@dataclass
class RolloutBuffer:
    grid: np.ndarray
    base_feats: np.ndarray
    raw_agent: np.ndarray
    raw_base: np.ndarray
    scalar: np.ndarray
    gstate: np.ndarray       # encoded global-state planes (centralized critic)
    gscalar: np.ndarray      # encoded global-state scalars (centralized critic)
    actions: np.ndarray
    logprobs: np.ndarray
    rewards: np.ndarray      # reward fed to GAE (shaped, if a shaper is set)
    env_rewards: np.ndarray  # unshaped env-default reward (for true-return logging)
    dones: np.ndarray
    masks: np.ndarray

    @property
    def size(self):
        return self.grid.shape[0]


def _new_buffer(n):
    return RolloutBuffer(
        grid=np.zeros((n, GRID_CHANNELS, GRID_SIZE, GRID_SIZE), np.float32),
        base_feats=np.zeros((n, NUM_BASES, BASE_FIELDS), np.float32),
        raw_agent=np.zeros((n, *RAW_AGENT_SHAPE), np.float32),
        raw_base=np.zeros((n, *RAW_BASE_SHAPE), np.float32),
        scalar=np.zeros((n, FEATURE_SCALARS), np.float32),
        gstate=np.zeros((n, STATE_PLANES, GRID_SIZE, GRID_SIZE), np.float32),
        gscalar=np.zeros((n, STATE_SCALARS), np.float32),
        actions=np.zeros(n, np.int64),
        logprobs=np.zeros(n, np.float32),
        rewards=np.zeros(n, np.float32),
        env_rewards=np.zeros(n, np.float32),
        dones=np.zeros(n, np.float32),
        masks=np.zeros((n, NUM_ACTIONS), bool),
    )


def collect_rollout(actor, registry, learner_slots, num_episodes, seed0,
                    opponent_members=None, reward_shaper=None,
                    return_outcomes=False):
    """Roll out `num_episodes` episodes; collect transitions from every
    learner slot. Returns a RolloutBuffer, or (RolloutBuffer, outcomes) when
    return_outcomes=True.

    outcomes: list of (Member, learner_won) — one entry per (episode,
        opponent slot). learner_won is True when the learner's mean per-slot
        episode score (over learner_slots) exceeds that opponent slot's
        episode score. Members that are the string 'random' are skipped
        (not league Members — they carry no bookkeeping).

    opponent_members: list of league Members (or 'random') to fill non-
        learner slots; one is sampled per slot per episode.
    reward_shaper: optional callable(reward, action, step) -> shaped_reward.
        None -> env default reward used unchanged.
    """
    cfg = default_config()
    cfg.env.novice = True
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    dynamics = env.unwrapped.dynamics   # entity registry for the critic's global state
    learner_slots = tuple(learner_slots)
    n = len(learner_slots) * EPISODE_LEN * num_episodes
    buf = _new_buffer(n)
    actor.eval()
    device = next(actor.parameters()).device   # feed rollout inputs to the actor's device
    total_written = 0
    outcomes = []   # (Member, learner_won) per (episode, opponent slot)

    for ep in range(num_episodes):
        seed = seed0 + ep
        random.seed(seed)
        env.reset(seed=seed)
        ep_scores = {s: 0.0 for s in SLOTS}   # env-default reward per slot
        # opponents fill the non-learner slots
        opp_slots = [s for s in SLOTS if s not in learner_slots]
        opp_agents = {}
        opp_member = {}        # slot -> the league Member (or 'random')
        for s in opp_slots:
            member = "random" if not opponent_members else random.choice(
                opponent_members)
            ag = registry.make(member)
            ag.reset()
            opp_agents[s] = ag
            opp_member[s] = member
        # per-learner-slot feature builders + last-transition bookkeeping
        fbs = {s: FeatureBuilder() for s in learner_slots}
        # run-contiguous base index: episode ep, slot at position j in learner_slots
        # occupies buf indices [base : base + EPISODE_LEN]
        slot_base = {s: (ep * len(learner_slots) + j) * EPISODE_LEN
                     for j, s in enumerate(learner_slots)}
        # per-slot write cursor within the run (0 .. EPISODE_LEN-1)
        slot_cursor = {s: 0 for s in learner_slots}
        # last buffer index written for each slot (for reward/done back-fill)
        prev_idx = {s: None for s in learner_slots}

        for slot in env.agent_iter():
            obs, reward, term, trunc, _ = env.last()
            ep_scores[slot] += float(reward)
            done = term or trunc
            if slot in learner_slots:
                step = int(np.asarray(obs["step"]).flat[0])
                if prev_idx[slot] is not None:
                    r = float(reward)
                    buf.env_rewards[prev_idx[slot]] = r
                    if reward_shaper is not None:
                        r = reward_shaper(r, buf.actions[prev_idx[slot]], step)
                    buf.rewards[prev_idx[slot]] = r
                    buf.dones[prev_idx[slot]] = 1.0 if done else 0.0
                if done:
                    env.step(None)
                    prev_idx[slot] = None
                    continue
                grid, base_feats, raw_agent, raw_base, scalar = \
                    fbs[slot].build(obs)
                mask = np.asarray(obs["action_mask"], dtype=bool).reshape(-1)
                with torch.no_grad():
                    a, logp, _ = actor.act(
                        torch.from_numpy(grid).unsqueeze(0).to(device),
                        torch.from_numpy(base_feats).unsqueeze(0).to(device),
                        torch.from_numpy(raw_agent).unsqueeze(0).to(device),
                        torch.from_numpy(raw_base).unsqueeze(0).to(device),
                        torch.from_numpy(scalar).unsqueeze(0).to(device),
                        torch.from_numpy(mask).unsqueeze(0).to(device),
                    )
                # run-contiguous destination index
                dest = slot_base[slot] + slot_cursor[slot]
                gp, gsc = encode_global_state(dynamics, slot, step)
                buf.grid[dest] = grid
                buf.base_feats[dest] = base_feats
                buf.raw_agent[dest] = raw_agent
                buf.raw_base[dest] = raw_base
                buf.scalar[dest] = scalar
                buf.gstate[dest] = gp
                buf.gscalar[dest] = gsc
                buf.actions[dest] = int(a[0])
                buf.logprobs[dest] = float(logp[0])
                buf.masks[dest] = mask
                prev_idx[slot] = dest
                slot_cursor[slot] += 1
                total_written += 1
                env.step(int(a[0]))
            else:
                if done:
                    env.step(None)
                else:
                    env.step(opp_agents[slot].action(obs))
        # --- per-opponent-slot outcomes for league.record_result (§6.2) ---
        learner_mean = (sum(ep_scores[s] for s in learner_slots)
                        / len(learner_slots))
        for s in opp_slots:
            member = opp_member[s]
            if member == "random":
                continue   # not a league Member; no bookkeeping
            learner_won = learner_mean > ep_scores[s]
            outcomes.append((member, learner_won))
    env.close()
    assert total_written == n, f"collected {total_written}, expected {n}"
    if return_outcomes:
        return buf, outcomes
    return buf


def _concat_buffers(bufs):
    """Concatenate RolloutBuffers field-wise, preserving order."""
    return RolloutBuffer(
        grid=np.concatenate([b.grid for b in bufs]),
        base_feats=np.concatenate([b.base_feats for b in bufs]),
        raw_agent=np.concatenate([b.raw_agent for b in bufs]),
        raw_base=np.concatenate([b.raw_base for b in bufs]),
        scalar=np.concatenate([b.scalar for b in bufs]),
        gstate=np.concatenate([b.gstate for b in bufs]),
        gscalar=np.concatenate([b.gscalar for b in bufs]),
        actions=np.concatenate([b.actions for b in bufs]),
        logprobs=np.concatenate([b.logprobs for b in bufs]),
        rewards=np.concatenate([b.rewards for b in bufs]),
        env_rewards=np.concatenate([b.env_rewards for b in bufs]),
        dones=np.concatenate([b.dones for b in bufs]),
        masks=np.concatenate([b.masks for b in bufs]),
    )


def _worker_rollout(state_dict, cfg, learner_slots, num_episodes, seed0,
                    opponent_members, reward_shaper):
    """Pool-worker entry point: rebuild a SymbolicTransformerActor from the CPU
    `state_dict` and `cfg` supplied by the caller and run `num_episodes` episodes
    via the unchanged collect_rollout. Top-level + picklable so multiprocessing
    can dispatch it. Returns (RolloutBuffer, outcomes)."""
    torch.set_num_threads(1)   # one thread per worker — avoid core oversubscription
    random.seed(seed0)
    np.random.seed(seed0)
    torch.manual_seed(seed0)
    actor = SymbolicTransformerActor(**cfg)
    actor.load_state_dict(state_dict)
    actor.eval()
    registry = OpponentRegistry(league=None)   # make() never touches the league
    return collect_rollout(actor, registry, learner_slots, num_episodes, seed0,
                           opponent_members=opponent_members,
                           reward_shaper=reward_shaper, return_outcomes=True)


def _worker_rollout_star(args):
    """imap-friendly single-arg adapter for _worker_rollout — lets the
    progress-bar path dispatch with order-preserving pool.imap."""
    return _worker_rollout(*args)


def collect_rollout_parallel(actor, learner_slots, num_episodes, seed0,
                             opponent_members, reward_shaper, pool, num_workers,
                             progress=True):
    """Parallel episode rollout: split `num_episodes` into contiguous chunks,
    run collect_rollout per chunk across `num_workers` pool workers, and
    concatenate. Episode `ep` keeps seed `seed0 + ep` regardless of chunking.
    `pool` is a caller-owned multiprocessing.Pool (its lifecycle is the
    caller's responsibility).

    progress: when True (default), dispatch one episode per task and show a
    tqdm bar as tasks complete. Pass False to chunk by worker count and
    dispatch silently. Returns (RolloutBuffer, outcomes)."""
    workers = max(1, min(num_workers, num_episodes))
    n_chunks = num_episodes if progress else workers
    base, extra = divmod(num_episodes, n_chunks)
    state_dict = {k: v.cpu() for k, v in actor.state_dict().items()}
    cfg = actor.cfg
    tasks = []
    offset = 0
    for i in range(n_chunks):
        count = base + (1 if i < extra else 0)
        if count == 0:
            continue
        tasks.append((state_dict, cfg, learner_slots, count, seed0 + offset,
                      opponent_members, reward_shaper))
        offset += count
    if progress:
        from tqdm.auto import tqdm
        results = list(tqdm(pool.imap(_worker_rollout_star, tasks),
                            total=len(tasks), desc="rollout episodes"))
    else:
        results = pool.starmap(_worker_rollout, tasks)
    bufs = [buf for buf, _ in results]
    outcomes = [o for _, oc in results for o in oc]
    return _concat_buffers(bufs), outcomes


def compute_advantages(rewards, values, dones, gamma, gae_lambda):
    """GAE over a flat multi-slot buffer.

    Each learner slot's transitions form a contiguous run; dones[t]==1 marks an
    episode end (and thus a slot-run end), which zeroes the bootstrap. The step
    after the final transition is treated as terminal.
    Returns (advantages, returns), float32 arrays of shape (T,).
    """
    rewards = np.asarray(rewards, np.float32)
    values = np.asarray(values, np.float32)
    dones = np.asarray(dones, np.float32)
    T = rewards.shape[0]
    adv = np.zeros(T, np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_nonterminal = 0.0
            next_value = 0.0
        else:
            next_nonterminal = 1.0 - dones[t]
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        adv[t] = last_gae
    returns = adv + values
    return adv, returns


def critic_values(critic, gstate, gscalar, device):
    """Run the centralized critic over a batch of encoded global states."""
    with torch.no_grad():
        v = critic(torch.from_numpy(gstate).to(device),
                   torch.from_numpy(gscalar).to(device)).cpu().numpy()
    return v.astype(np.float32)


import torch.nn as nn
import torch.optim as optim


@dataclass
class PPOConfig:
    learning_rate: float = 1e-5      # KNOB
    num_minibatches: int = 8         # KNOB — bumped from 4: 4 minibatches OOM'd
                                     # on a T4 (the 347-token transformer's
                                     # attention is [mb, h, T, T] per layer
                                     # per actor+critic; 4 -> mb=600 blew
                                     # 14GB, 16 -> mb=150 fits comfortably).
    update_epochs: int = 4           # KNOB
    clip_coef: float = 0.2           # KNOB
    ent_coef: float = 0.01           # KNOB — start value; annealed to ent_coef_final
    ent_coef_final: float = 0.01     # KNOB — ent_coef linearly anneals to this by the last update
    vf_coef: float = 0.5             # KNOB
    max_grad_norm: float = 0.5       # KNOB


def make_optimizer(actor, critic, cfg):
    """Adam over the actor + centralized-critic parameters.

    Build one per training run and thread it through ppo_update so its Adam
    momentum state persists across updates — and dies with the run. (The old
    module-global cache keyed by id() could alias a freed actor's optimizer
    onto a new one after id reuse, and leaked across tests.)
    """
    return optim.Adam(list(actor.parameters()) + list(critic.parameters()),
                      lr=cfg.learning_rate, eps=1e-5)


def ppo_update(actor, critic, opt, buf, advantages, returns, cfg, device):
    """One PPO update over the rollout buffer. Trains actor + centralized
    critic via the caller-owned optimizer `opt` (see make_optimizer)."""
    n = buf.size
    grid = torch.from_numpy(buf.grid).to(device)
    base_feats = torch.from_numpy(buf.base_feats).to(device)
    raw_agent = torch.from_numpy(buf.raw_agent).to(device)
    raw_base = torch.from_numpy(buf.raw_base).to(device)
    scalar = torch.from_numpy(buf.scalar).to(device)
    masks = torch.from_numpy(buf.masks).to(device)
    actions = torch.from_numpy(buf.actions).to(device)
    old_logp = torch.from_numpy(buf.logprobs).to(device)
    gstate = torch.from_numpy(buf.gstate).to(device)
    gscalar = torch.from_numpy(buf.gscalar).to(device)
    adv = torch.from_numpy(np.asarray(advantages, np.float32)).to(device)
    ret = torch.from_numpy(np.asarray(returns, np.float32)).to(device)

    mb_size = max(1, n // cfg.num_minibatches)
    inds = np.arange(n)
    actor.train()
    critic.train()
    pg = vl = ent = torch.zeros((), device=device)
    for _ in range(cfg.update_epochs):
        np.random.shuffle(inds)
        for start in range(0, n, mb_size):
            mb = inds[start:start + mb_size]
            mbt = torch.from_numpy(mb).to(device)
            _, newlogp, entropy = actor.act(
                grid[mbt], base_feats[mbt], raw_agent[mbt], raw_base[mbt],
                scalar[mbt], masks[mbt], action=actions[mbt])
            ratio = (newlogp - old_logp[mbt]).exp()
            a = adv[mbt]
            a = (a - a.mean()) / (a.std() + 1e-8)
            pg1 = -a * ratio
            pg2 = -a * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
            pg = torch.max(pg1, pg2).mean()
            newvalue = critic(gstate[mbt], gscalar[mbt])
            vl = 0.5 * ((newvalue - ret[mbt]) ** 2).mean()
            ent = entropy.mean()
            loss = pg - cfg.ent_coef * ent + cfg.vf_coef * vl
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()),
                cfg.max_grad_norm)
            opt.step()
    return {"policy_loss": float(pg.detach()), "value_loss": float(vl.detach()),
            "entropy": float(ent.detach())}


import time

from scripted.geometry import STAY


class AntiIdleShaper:
    """Training-only reward shaping: a small penalty on the STAY action,
    linearly annealed to zero by the end of training.

    The original PPO collapsed to a flat-0 idle policy; this is the backstop
    (BC warm-start is the primary defence). The eval path NEVER constructs a
    shaper — evaluate.py always scores the unmodified default reward.
    """

    def __init__(self, penalty=0.05, total_updates=1000):  # KNOB: penalty
        self.penalty = penalty
        self.total_updates = max(1, total_updates)
        self._frac = 1.0          # 1.0 = full strength, 0.0 = annealed off

    def set_update(self, update):
        """Set the current training update; recomputes the anneal fraction."""
        self._frac = max(0.0, 1.0 - update / self.total_updates)

    def __call__(self, reward, action, step):
        """Shape one transition. STAY is penalized; everything else passes."""
        if int(action) == STAY:
            return reward - self.penalty * self._frac
        return reward


class RungLadder:
    """The 3-rung self-play ladder.

    Rung 1: opponents = the 6 scripted strategies (deterministic anchors).
    Rung 2: opponents = PFSP-sampled frozen checkpoints.
    Rung 3: live self-play — recent checkpoints (rung 2 pool still in play).
    Promotion: advance when the learner's win-rate vs the CURRENT rung's
    opponents clears `promote_winrate` over an eval batch.
    """

    def __init__(self, league, promote_winrate=0.7):  # KNOB: promote_winrate
        self.league = league
        self.promote_winrate = promote_winrate
        self.rung = 1

    def current_opponents(self):
        """League Members the rollout should draw opponents from this rung."""
        if self.rung == 1:
            return self.league.anchors()
        # rungs 2 and 3 both draw from the checkpoint pool; rung 3 additionally
        # includes the freshest checkpoints. If the pool is empty, fall back to
        # anchors so training never stalls.
        ckpts = self.league.checkpoints()
        return ckpts if ckpts else self.league.anchors()

    def sample_opponents(self):
        """PFSP-sample opponents for the current rung (anchors at rung 1 are
        sampled uniformly; rungs 2-3 use the league's PFSP weights)."""
        if self.rung == 1:
            return self.league.anchors()
        return [self.league.sample_opponent()
                for _ in range(len(SLOTS) - 1)]

    def try_promote(self, win_rate):
        """Advance a rung if win_rate clears the gate. Returns True on promote."""
        if self.rung >= 3:
            return False
        if win_rate >= self.promote_winrate:
            self.rung += 1
            return True
        return False


def _build_viz_slot_agents(actor, learner_slots, opponents, update):
    """Build a render_episode slot_agents dict from the current matchup.

    The learner (a NeuralAgent over a CPU copy of `actor`) fills
    learner_slots; `opponents` (league Members) fill the remaining slots in
    order. Returns {slot: (agent, label)} for all six slots.
    """
    import copy

    learner_slots = tuple(learner_slots)
    opp_slots = [s for s in SLOTS if s not in learner_slots]
    # a frozen CPU copy so the replay never perturbs the training actor
    cpu_actor = copy.deepcopy(actor).to("cpu")
    cpu_actor.eval()

    slot_agents = {}
    learner_label = f"learner @ upd{update}"
    for s in learner_slots:
        slot_agents[s] = (NeuralAgent(cpu_actor, name=learner_label),
                          learner_label)
    registry = OpponentRegistry(League())   # stateless factory for agents
    for i, s in enumerate(opp_slots):
        member = opponents[i % len(opponents)]
        agent = registry.make(member)
        label = member.name if hasattr(member, "name") else "random"
        slot_agents[s] = (agent, label)
    return slot_agents


@dataclass
class Args:
    total_updates: int = 1000                  # KNOB
    episodes_per_update: int = 4               # KNOB
    rollout_workers: int = 4                   # KNOB — parallel rollout worker processes
    learner_slots: tuple = ("agent_0", "agent_1", "agent_2")  # KNOB
    gamma: float = 0.99
    gae_lambda: float = 0.95
    snapshot_every: int = 25                   # KNOB
    eval_every: int = 25                       # KNOB
    eval_seeds: int = 16                       # KNOB
    viz_every: int = 25                        # KNOB: 0 disables the viz hook
    anti_idle_penalty: float = 0.05            # KNOB
    bc_init: str = "policy_bc.pt"
    critic_init: str = "critic_pretrained.pt"  # KNOB — pretrained critic in
                                               # ae/training; "" -> random critic
    seed: int = 1
    cuda: bool = True


def main():
    import tyro
    from evaluate import evaluate_policy
    args = tyro.cli(Args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available()
                           else "cpu")
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "..", "src")
    from metrics import MetricsLogger
    from visualize import render_episode
    VIZ_DIR = os.path.join(here, "viz")
    os.makedirs(VIZ_DIR, exist_ok=True)
    logger = MetricsLogger(VIZ_DIR) if args.viz_every else None

    # Created before the actor touches CUDA so the forked workers inherit a
    # CUDA-free parent (workers do CPU-only inference). Pool size is capped at
    # episodes_per_update — no point spawning more workers than episodes.
    rollout_pool = multiprocessing.Pool(
        min(args.rollout_workers, args.episodes_per_update))
    bc_path = os.path.join(src, args.bc_init)
    if os.path.exists(bc_path):
        actor = SymbolicTransformerActor.from_checkpoint(bc_path).to(device)
        print(f"actor warm-started from {bc_path}")
    else:
        actor = SymbolicTransformerActor().to(device)
        print(f"WARNING: bc_init {bc_path} not found — actor trained from "
              f"SCRATCH (random init). This will almost certainly collapse; "
              f"pass --bc-init <existing checkpoint in ae/src>.")
    critic = CentralizedCritic().to(device)
    if args.critic_init:
        critic_path = os.path.join(here, args.critic_init)
        if os.path.exists(critic_path):
            ck = torch.load(critic_path, map_location=device)
            critic = CentralizedCritic(**ck["cfg"]).to(device)
            critic.load_state_dict(ck["state_dict"])
            print(f"critic warm-started from {critic_path}")
        else:
            print(f"critic_init {critic_path} not found — random critic")

    league = League()
    registry = OpponentRegistry(league)
    ladder = RungLadder(league)
    shaper = AntiIdleShaper(args.anti_idle_penalty, args.total_updates)
    cfg = PPOConfig()
    opt = make_optimizer(actor, critic, cfg)   # one optimizer for the whole run
    ent_coef_start = cfg.ent_coef          # ent_coef anneals start -> ent_coef_final
    best_winrate = -1.0

    from tqdm.auto import tqdm
    pbar = tqdm(range(1, args.total_updates + 1), desc="ppo")
    for update in pbar:
        shaper.set_update(update)
        frac = (update - 1) / max(1, args.total_updates - 1)
        cfg.ent_coef = ent_coef_start + frac * (cfg.ent_coef_final - ent_coef_start)
        opponents = ladder.sample_opponents()
        buf, outcomes = collect_rollout_parallel(
            actor, args.learner_slots, args.episodes_per_update,
            seed0=random.randint(0, 2**30),
            opponent_members=opponents, reward_shaper=shaper,
            pool=rollout_pool, num_workers=args.rollout_workers)
        # outcomes' Member objects are pickled COPIES returned from the
        # rollout worker processes — recording on them would never touch the
        # league's real Members, leaving every winrate a flat 0.5 (games==0)
        # and PFSP sampling silently uniform. Re-map to the canonical Member
        # by name (unique: "scripted:<strat>" / "ckpt:<update>").
        by_name = {m.name: m for m in league.members()}
        for member, learner_won in outcomes:
            league.record_result(by_name.get(member.name, member), learner_won)
        values = critic_values(critic, buf.gstate, buf.gscalar, device)
        adv, ret = compute_advantages(buf.rewards, values, buf.dones,
                                      args.gamma, args.gae_lambda)
        losses = ppo_update(actor, critic, opt, buf, adv, ret, cfg, device)

        # Per-update progress: rung, mean env-default return per (episode,slot),
        # current policy loss, annealed entropy coefficient.
        _n_ret = args.episodes_per_update * len(args.learner_slots)
        _mean_return = float(buf.env_rewards.sum()) / max(1, _n_ret)
        pbar.set_postfix(rung=ladder.rung,
                         ret=f"{_mean_return:.1f}",
                         loss=f"{losses['policy_loss']:.3f}",
                         ent=f"{cfg.ent_coef:.3f}")

        if logger is not None:
            # true env-default return, averaged per (episode, learner slot) —
            # NOT the shaped buf.rewards (which the anti-idle shaper perturbs).
            n_returns = args.episodes_per_update * len(args.learner_slots)
            mean_return = float(buf.env_rewards.sum()) / max(1, n_returns)
            metrics = {
                "policy_loss": losses["policy_loss"],
                "value_loss": losses["value_loss"],
                "entropy": losses["entropy"],
                "mean_return": mean_return,
                "rung": ladder.rung,
                "pool_size": len(league.checkpoints()),
                "anti_idle_coef": shaper.penalty * shaper._frac,
                "ent_coef": cfg.ent_coef,
            }

        if update % args.snapshot_every == 0:
            ckpt = os.path.join(src, f"policy_rung{ladder.rung}_u{update}.pt")
            actor.save_checkpoint(ckpt)
            league.snapshot(ckpt, update)
            actor.save_checkpoint(os.path.join(src, "policy_final.pt"))

        if update % args.eval_every == 0:
            opp_specs = ladder.current_opponents()
            opp_agents = [registry.make(random.choice(opp_specs))
                          for _ in range(5)]
            # evaluate on a frozen CPU copy — never move the training actor
            # (its optimizer state is pinned to `device`).
            eval_actor = copy.deepcopy(actor).to("cpu")
            res = evaluate_policy(NeuralAgent(eval_actor, "learner"),
                                  opp_agents, list(range(args.eval_seeds)))
            print(f"update {update} rung {ladder.rung} "
                  f"winrate={res.win_rate:.2f} score={res.mean_score:.1f} "
                  f"{losses}")
            if res.win_rate > best_winrate:
                best_winrate = res.win_rate
                actor.save_checkpoint(
                    os.path.join(src, f"policy_best_rung{ladder.rung}.pt"))
            if ladder.try_promote(res.win_rate):
                print(f"PROMOTED to rung {ladder.rung}")
            if logger is not None:
                metrics["eval_winrate"] = res.win_rate
                metrics["eval_score"] = res.mean_score

        if logger is not None:
            logger.log(update, metrics)
            if update % args.viz_every == 0:
                try:
                    logger.plot(os.path.join(VIZ_DIR, f"metrics_u{update}.png"))
                    logger.leaderboard(
                        league, update,
                        os.path.join(VIZ_DIR, "leaderboard.csv"),
                        os.path.join(VIZ_DIR, f"leaderboard_u{update}.png"))
                    slot_agents = _build_viz_slot_agents(
                        actor, args.learner_slots, opponents, update)
                    render_episode(
                        slot_agents,
                        os.path.join(VIZ_DIR, f"replay_u{update}.mp4"),
                        fps=5, max_steps=EPISODE_LEN, seed=update)
                except Exception as e:
                    print(f"[viz] update {update}: visualization failed, continuing: {e}")

    rollout_pool.close()
    rollout_pool.join()
    if logger is not None:
        logger.close()
    actor.save_checkpoint(os.path.join(src, "policy_final.pt"))
    print("self-play training done -> policy_final.pt")


if __name__ == "__main__":
    main()
