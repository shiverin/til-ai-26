"""League pool: scripted anchors are permanent; snapshots accumulate."""
from league import League


def test_six_scripted_anchors_present():
    lg = League()
    names = {m.name for m in lg.members()}
    for strat in ("balanced", "balanced_extreme", "base_rusher",
                  "base_rusher_extreme", "collector", "camper"):
        assert f"scripted:{strat}" in names


def test_snapshot_adds_a_checkpoint_member(tmp_path):
    lg = League()
    n0 = len(lg.members())
    lg.snapshot(str(tmp_path / "ckpt_0.pt"), update=10)
    assert len(lg.members()) == n0 + 1
    assert any(m.kind == "checkpoint" for m in lg.members())


def test_anchors_never_removed_when_pool_capped(tmp_path):
    lg = League(max_checkpoints=2)
    for i in range(5):
        lg.snapshot(str(tmp_path / f"ckpt_{i}.pt"), update=i)
    ckpts = [m for m in lg.members() if m.kind == "checkpoint"]
    anchors = [m for m in lg.members() if m.kind == "scripted"]
    assert len(ckpts) == 2          # pool capped
    assert len(anchors) == 6        # anchors untouched
