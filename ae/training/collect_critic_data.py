"""Collect frozen-actor rollouts for centralized-critic pretraining.

Stage between BC and PPO. Runs the BC-cloned actor (frozen) against the fixed
scripted opponent panel and dumps, for every learner transition, the
centralized critic's global-state encoding plus the env-default reward stream.
pretrain_critic.py turns the reward stream into reward-to-go targets and fits
CentralizedCritic by MSE — so PPO starts with a calibrated critic instead of a
random one (a random critic gives garbage advantages and destabilizes the
policy gradient immediately).

The rollout reuses train_selfplay.collect_rollout_parallel with no reward
shaper, so `env_rewards` carries the unshaped env-default reward on the same
scale PPO's GAE consumes.
"""
import argparse
import multiprocessing
import os
import sys
import time

import torch

# ae/src holds scripted/, policy.py — needed before importing train_selfplay's
# dependencies (it adds the path itself, but league/policy imports here race it).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from league import League
from policy import SymbolicTransformerActor
from train_selfplay import EPISODE_LEN, collect_rollout_parallel


def main():
    parser = argparse.ArgumentParser(
        description="Collect frozen-actor rollouts for critic pretraining")
    parser.add_argument("--actor", default="policy_family_winner_bc.pt",
                        help="BC actor checkpoint in ae/src "
                             "(default: policy_family_winner_bc.pt)")
    parser.add_argument("--episodes", type=int, default=200,
                        help="rollout episodes (default 200)")
    parser.add_argument("--learner-slots", nargs="+",
                        default=["agent_0", "agent_1", "agent_2"],
                        help="slots the frozen actor controls")
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel rollout worker processes")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="logs/critic_pretrain_data.pt",
                        help="output path, relative to ae/training")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    actor_path = os.path.join(here, "..", "src", args.actor)
    actor = SymbolicTransformerActor.from_checkpoint(actor_path)
    actor.eval()   # frozen — never updated; CPU-resident for the fork workers
    print(f"frozen actor loaded from {actor_path}")

    # fixed scripted opponent panel — the 6 strategy anchors, sampled per slot
    opponents = League().anchors()

    t0 = time.time()
    pool = multiprocessing.Pool(min(args.workers, args.episodes))
    try:
        buf, _ = collect_rollout_parallel(
            actor, tuple(args.learner_slots), args.episodes, seed0=args.seed,
            opponent_members=opponents, reward_shaper=None,
            pool=pool, num_workers=args.workers, progress=True)
    finally:
        pool.close()
        pool.join()

    out_path = os.path.join(here, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({
        "gstate": buf.gstate,
        "gscalar": buf.gscalar,
        "env_rewards": buf.env_rewards,
        "dones": buf.dones,
        "episode_len": EPISODE_LEN,
        "meta": {
            "actor": args.actor,
            "episodes": args.episodes,
            "learner_slots": list(args.learner_slots),
            "transitions": int(buf.size),
            "duration_seconds": time.time() - t0,
        },
    }, out_path)
    print(f"saved {buf.size} transitions -> {out_path} "
          f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
