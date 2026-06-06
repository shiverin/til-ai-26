from scripted.belief import Belief
from scripted.handover import HandoverTrigger
from scripted.map_prior import MapPrior


def _fresh_belief():
    prior = MapPrior.load()
    prior.identify_team(prior.bases[0])
    b = Belief()
    b.reset(prior)
    return b


def test_does_not_fire_early_with_no_destroyed_bases():
    b = _fresh_belief()
    b.step = 10
    b.dead_bases = set()
    assert HandoverTrigger()(b) is False


def test_fires_on_first_destroyed_enemy_base():
    b = _fresh_belief()
    b.step = 10
    # arbitrary cell; HandoverTrigger only counts len(dead_bases)
    b.dead_bases = {(7, 7)}
    assert HandoverTrigger()(b) is True


def test_fires_at_step_fallback_even_with_no_destroyed_bases():
    b = _fresh_belief()
    b.step = 60
    b.dead_bases = set()
    assert HandoverTrigger()(b) is True


def test_does_not_fire_one_step_before_fallback():
    b = _fresh_belief()
    b.step = 59          # step_fallback - 1, empty dead_bases
    b.dead_bases = set()
    assert HandoverTrigger()(b) is False


def test_custom_thresholds():
    b = _fresh_belief()
    b.step = 5
    b.dead_bases = {(7, 7)}
    assert HandoverTrigger(min_destroyed_enemy_bases=2, step_fallback=100)(b) is False


def test_custom_threshold_met():
    b = _fresh_belief()
    b.step = 5
    b.dead_bases = {(7, 7), (8, 8)}    # two destroyed, threshold 2 → fires
    assert HandoverTrigger(min_destroyed_enemy_bases=2, step_fallback=100)(b) is True
