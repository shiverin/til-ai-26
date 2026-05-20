from scripted.strategies import StrategyParams


def test_strategy_params_defaults():
    p = StrategyParams()
    assert p.breach_min_bombs == 2
    assert p.sweep_base_gradient == 0.5
    assert p.forage_requires_endgame is True
    assert p.camp_leash is None


def test_strategy_params_is_frozen():
    p = StrategyParams()
    try:
        p.camp_leash = 4
        assert False, "StrategyParams must be frozen"
    except AttributeError:
        pass


def test_strategy_params_override():
    p = StrategyParams(camp_leash=4, forage_requires_endgame=False)
    assert p.camp_leash == 4
    assert p.forage_requires_endgame is False
    assert p.sweep_base_gradient == 0.5  # untouched default


def test_strategy_registry_has_layer_only_strategies():
    from scripted.strategies import STRATEGIES
    for name in ("balanced", "balanced_extreme", "base_rusher",
                 "base_rusher_extreme", "collector"):
        s = STRATEGIES[name]
        assert s.name == name
        assert len(s.layers) >= 3
        assert all(callable(layer) for layer in s.layers)


def test_balanced_strike_order():
    from scripted.strategies import STRATEGIES
    from scripted.layers import hunt, survive, strike
    bal = STRATEGIES["balanced"].layers
    ext = STRATEGIES["balanced_extreme"].layers
    # balanced: survive first, then hunt, then strike.
    assert bal.index(survive) < bal.index(hunt) < bal.index(strike)
    # balanced_extreme: hunt first, then strike, with strike still before
    # survive (the "extreme" identity — bombs a base from a dangerous cell).
    assert ext.index(hunt) < ext.index(strike) < ext.index(survive)


def test_base_rusher_uses_default_params():
    from scripted.strategies import STRATEGIES
    assert STRATEGIES["base_rusher"].params == StrategyParams()
    assert STRATEGIES["base_rusher_extreme"].params == StrategyParams()


def test_act_runs_with_explicit_strategy():
    from scripted.belief import Belief
    from scripted.decide import act
    from scripted.map_prior import MapPrior
    from scripted.strategies import STRATEGIES
    prior = MapPrior.load()
    prior.identify_team(prior.bases[0])
    b = Belief()
    b.reset(prior)
    b.location = prior.spawns[0]["pos"]
    b.facing = prior.spawns[0]["facing"]
    a = act(b, [1, 1, 1, 1, 1, 1], STRATEGIES["base_rusher"])
    assert 0 <= a <= 5


def test_registry_has_all_strategies():
    from scripted.strategies import STRATEGIES
    assert set(STRATEGIES) == {
        "balanced", "balanced_extreme", "base_rusher", "base_rusher_extreme",
        "collector", "camper", "forager", "lean_rush", "defender",
    }
    for name, s in STRATEGIES.items():
        assert s.name == name
        assert len(s.layers) >= 2
        assert all(callable(layer) for layer in s.layers)


def test_camper_params_and_layers():
    from scripted.strategies import STRATEGIES
    from scripted.layers import camp, hold
    camper = STRATEGIES["camper"]
    assert camper.params.camp_leash == 4
    assert camper.params.forage_requires_endgame is False
    assert camp in camper.layers
    assert camper.layers[-1] is hold  # hold is the final fallback


def test_balanced_includes_hunt():
    from scripted.strategies import STRATEGIES
    from scripted.layers import hunt
    assert hunt in STRATEGIES["balanced"].layers
    assert hunt in STRATEGIES["balanced_extreme"].layers


def test_strategy_params_has_offense_knobs():
    p = StrategyParams()
    assert p.target_travel_weight == 0.05
    assert p.soften_floor == 60.0


def test_forager_strategy_is_registered():
    from scripted.strategies import STRATEGIES
    from scripted.adaptive_layers import forage_loop
    from scripted.layers import survive, sweep, default
    s = STRATEGIES["forager"]
    assert s.name == "forager"
    assert s.layers == (survive, forage_loop, sweep, default)


def test_strategyparams_has_roi_offense_knobs():
    from scripted.strategies import StrategyParams
    p = StrategyParams()
    assert p.roi_gate_margin == 0.15
    assert p.vulture_hp_boost == 2.0


def test_lean_rush_strategy_is_registered():
    from scripted.strategies import STRATEGIES
    from scripted.adaptive_layers import rush_roi
    from scripted.layers import survive, hunt, sweep, default
    s = STRATEGIES["lean_rush"]
    assert s.name == "lean_rush"
    assert s.layers == (survive, hunt, rush_roi, sweep, default)


def test_strategyparams_has_defend_radius():
    from scripted.strategies import StrategyParams
    assert StrategyParams().defend_radius == 4


def test_defender_strategy_is_registered():
    from scripted.strategies import STRATEGIES
    from scripted.adaptive_layers import defend_intercept, forage_loop
    from scripted.layers import survive, sweep, hold
    s = STRATEGIES["defender"]
    assert s.name == "defender"
    assert s.layers == (survive, defend_intercept, forage_loop, sweep, hold)
