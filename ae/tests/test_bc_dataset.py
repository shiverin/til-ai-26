"""DAgger rollout produces (feature, action) pairs labeled by the teacher."""
import numpy as np

from bc import collect_dagger_dataset, BCSample
from features import GRID_CHANNELS
from policy import SymbolicTransformerActor, NUM_ACTIONS


def test_teacher_only_dataset_shapes():
    """Beta=1.0: the teacher drives every step; dataset has correct shapes."""
    ds = collect_dagger_dataset(
        teacher_strategy="balanced",
        rollout_policy=None,       # beta=1.0 -> teacher controls
        beta=1.0,
        num_episodes=2,
        seeds=[0, 1],
    )
    assert len(ds) > 0
    s = ds[0]
    assert isinstance(s, BCSample)
    assert s.grid.shape == (GRID_CHANNELS, 16, 16)
    assert s.scalar.shape == (10,)
    assert 0 <= s.action < NUM_ACTIONS
    assert s.mask.shape == (NUM_ACTIONS,)


def test_actions_are_legal_under_mask():
    ds = collect_dagger_dataset("balanced", None, 1.0, 1, [3])
    for s in ds:
        assert s.mask[s.action]      # the teacher label is always a legal move


def test_mixed_rollout_runs():
    """Beta<1: a fresh actor drives some steps; labels still come from teacher."""
    actor = SymbolicTransformerActor()
    ds = collect_dagger_dataset("balanced", actor, beta=0.5, num_episodes=1,
                                seeds=[7])
    assert len(ds) > 0


def test_bcsample_carries_five_tensors():
    from bc import collect_dagger_dataset
    ds = collect_dagger_dataset("balanced", None, 1.0, 1, [0])
    s = ds[0]
    assert s.grid.shape == (17, 16, 16)
    assert s.base_feats.shape == (5, 11)
    assert s.raw_agent.shape == (7, 5, 25)
    assert s.raw_base.shape == (7, 7, 25)
    assert s.scalar.shape == (10,)
    assert s.mask.shape == (6,)


def test_dataset_contains_bomb_labels():
    """Regression (spec C Task 21): the teacher must demonstrate PLACE_BOMB.

    A dataset with zero bomb labels trains a clone structurally unable to bomb
    — it can never destroy an enemy base, the bulk of a strong score. This
    fails if collect_dagger_dataset drives the non-learner slots with the
    teacher (teacher-vs-teacher contention makes a competent strategy oscillate
    in place and never bomb); it passes with the varied opponent pool.
    """
    ds = collect_dagger_dataset(
        teacher_strategy="balanced",
        rollout_policy=None,
        beta=1.0,
        num_episodes=6,
        seeds=list(range(6)),
    )
    actions = {s.action for s in ds}
    assert 5 in actions, (
        f"no PLACE_BOMB (action 5) labels in dataset; actions present={sorted(actions)}"
    )
