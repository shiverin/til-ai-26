from scripted.belief import Belief
from scripted.danger import DangerMap
from scripted.layers import _openness, default, forage, hunt, strike, survive, sweep
from scripted.strategies import StrategyParams
from scripted.map_prior import MapPrior
from scripted.pathfind import build_planner


def _belief(loc, facing=0, team_bombs=3, bombs=None):
    m = MapPrior.load()
    m.identify_team((13, 9))                 # team 0
    b = Belief()
    b.reset(m)
    b.location = loc
    b.facing = facing
    b.team_bombs = team_bombs
    b.enemy_bombs = bombs or {}
    b.step = 10
    return b


def test_survive_returns_none_when_safe():
    b = _belief((5, 5))
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert survive(b, danger, p, StrategyParams()) is None


def test_survive_moves_off_a_dangerous_cell():
    # Bomb at (5,5) with timer=2 covers Chebyshev-2 radius at the exact
    # detonation tick. Every cell reachable in 1 step is within the blast;
    # every path out requires passing through the blast at tick 2. The
    # time-aware planner hard-blocks lethal (cell, tick) pairs, so no
    # cell outside the blast is reachable. Tier-2 selects (5,5) itself
    # (dist 0) as the "least-bad" candidate; survive correctly yields None
    # so objective layers can act (the agent will take the hit regardless).
    b = _belief((5, 5))
    danger = DangerMap({(5, 5): 2}, b)      # standing in a blast zone
    p = build_planner(b, danger)
    action = survive(b, danger, p, StrategyParams())
    assert action is None                    # inescapable blast — Tier 2 yields


def _base_prior(grid_size=7, wall_between=None, enemy_bases=((3, 3),)):
    class _Prior:
        pass
    p = _Prior()
    p.grid_size = grid_size
    p.wall_between = dict(wall_between or {})
    p.collectibles = {}
    p.enemy_bases = list(enemy_bases)
    p.our_base = (0, 0)
    return p


def _strike_belief(prior, loc, team_bombs=3):
    b = Belief()
    b.prior = prior
    b.destroyed_walls = set()
    b.dead_bases = set()
    b.enemy_base_health = {}
    b.ally_bombs = {}
    b.location = loc
    b.facing = 0
    b.team_bombs = team_bombs
    b.step = 10
    return b


def test_strike_places_bomb_in_range_of_enemy_base():
    """Strike bombs from within blast range + line-of-sight of a base, without
    having to stand on it."""
    b = _strike_belief(_base_prior(), (3, 5))   # 2 tiles from base (3,3), clear
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert strike(b, danger, p, StrategyParams()) == 5            # PLACE_BOMB


def test_strike_does_not_bomb_base_through_a_wall():
    """A base within blast range but behind a wall is not bomb-reachable —
    Strike navigates instead of wasting a bomb."""
    prior = _base_prior(wall_between={frozenset({(3, 5), (3, 4)}): False})
    b = _strike_belief(prior, (3, 5))
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert strike(b, danger, p, StrategyParams()) != 5            # cannot hit the base from here


def test_strike_skips_a_base_believed_destroyed():
    """A base believed destroyed is not targeted — with no live base, Strike
    yields to the next cascade layer."""
    b = _strike_belief(_base_prior(), (3, 5))
    b.dead_bases = {(3, 3)}                      # the only base is destroyed
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert strike(b, danger, p, StrategyParams()) is None


def test_strike_stops_bombing_a_doomed_base():
    """When the agent's own bombs already in flight will finish a base, Strike
    does not waste another bomb on it."""
    b = _strike_belief(_base_prior(), (3, 3))    # standing on the base
    b.enemy_base_health = {(3, 3): 0.4}          # 40 HP left
    b.own_bombs = [((3, 3), 3), ((3, 3), 3)]     # 2 own bombs reach it -> 40 dmg
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert strike(b, danger, p, StrategyParams()) is None          # base doomed


def _forage_belief(collectibles, loc, facing=0, enemy_bases=(), dead_bases=()):
    prior = _base_prior(grid_size=7, enemy_bases=list(enemy_bases))
    prior.collectibles = dict(collectibles)
    b = Belief()
    b.prior = prior
    b.destroyed_walls = set()
    b.dead_bases = set(dead_bases)
    b.collected = set()
    b.enemy_base_health = {}
    b.ally_bombs = {}
    b.location = loc
    b.facing = facing
    b.step = 10
    return b


def test_forage_inactive_while_a_base_lives():
    b = _forage_belief({(3, 3): 5.0}, (2, 2), enemy_bases=[(5, 5)])
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert forage(b, danger, p, StrategyParams()) is None          # a base still stands


def test_forage_steps_onto_a_collectible():
    # All bases gone; a collectible one tile ahead (facing RIGHT).
    b = _forage_belief({(3, 2): 5.0}, (2, 2), facing=0)
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert forage(b, danger, p, StrategyParams()) == 0             # FORWARD onto the loot


def test_forage_prefers_the_richer_two_move_path():
    # Facing RIGHT at (2,2): one low-value tile ahead, two high-value behind.
    collectibles = {(3, 2): 1.0, (1, 2): 5.0, (0, 2): 5.0}
    b = _forage_belief(collectibles, (2, 2), facing=0)
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert forage(b, danger, p, StrategyParams()) == 1             # BACKWARD — 10 value vs 1


def test_forage_yields_when_no_collectible_in_reach():
    b = _forage_belief({}, (2, 2))               # endgame, nothing to collect
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert forage(b, danger, p, StrategyParams()) is None          # falls through to sweep


def test_strike_holds_fire_with_no_bombs():
    b = _belief((9, 11), team_bombs=0)
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert strike(b, danger, p, StrategyParams()) != 5


def test_strike_places_bomb_without_escape_route():
    """Our own bomb is harmless to us, so Strike places a bomb whenever in
    range of an enemy base — no escape route required."""
    class _Prior:
        grid_size = 5            # on a 5x5 grid no tile is Chebyshev > 2 from
        wall_between = {}        # the centre, so an escape check could never pass
        collectibles = {}
        enemy_bases = [(2, 2)]
        our_base = (0, 0)

    b = Belief()
    b.prior = _Prior()
    b.destroyed_walls = set()
    b.location = (2, 2)          # standing on the enemy base
    b.facing = 0
    b.team_bombs = 3
    b.step = 10
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert strike(b, danger, p, StrategyParams()) == 5     # PLACE_BOMB — no escape needed


def test_openness_counts_open_area():
    # Fully open 5x5: BFS radius 4 from the centre reaches every cell.
    b = _strike_belief(_base_prior(grid_size=5, wall_between={}), (2, 2))
    danger = DangerMap({}, b)
    assert _openness(b, danger, (2, 2), radius=4) == 25


def test_openness_isolated_pocket_scores_one():
    # Wall (0,0) off from both its neighbours -> a one-cell pocket.
    prior = _base_prior(grid_size=5, wall_between={
        frozenset({(0, 0), (1, 0)}): False,
        frozenset({(0, 0), (0, 1)}): False,
    })
    b = _strike_belief(prior, (0, 0))
    danger = DangerMap({}, b)
    assert _openness(b, danger, (0, 0), radius=4) == 1        # walled-in pocket
    assert _openness(b, danger, (2, 2), radius=4) == 24       # everything but the pocket


# --- survive Tier 1: full escape, openness bias, opportunistic bomb drop ---

def test_survive_tier1_routes_to_safety():
    """One bomb, plenty of escape time, only 1 bomb in hand (below bomb_drop_min)
    -> survive routes toward safety with a move, no bomb drop."""
    b = _strike_belief(_base_prior(grid_size=5), (3, 2), team_bombs=1)
    danger = DangerMap({(1, 2): 4}, b)            # blast covers x0..3, agent at (3,2)
    p = build_planner(b, danger)
    assert survive(b, danger, p, StrategyParams()) == 0     # FORWARD toward x=4 safety


def test_survive_tier1_drops_surplus_bomb():
    """Same escape, but 3 bombs in hand and the chosen safe cell is 1 step away
    -> drop a bomb first (1 + 1 place + 1 buffer <= 4 ticks)."""
    b = _strike_belief(_base_prior(grid_size=5), (3, 2), team_bombs=3)
    danger = DangerMap({(1, 2): 4}, b)
    p = build_planner(b, danger)
    assert survive(b, danger, p, StrategyParams()) == 5     # PLACE_BOMB


def test_survive_tier1_no_bomb_without_spare_time():
    """3 bombs in hand but the agent is at the blast centre -> the nearest safe
    cell is 3 steps away, 3 + 1 + 1 > 4, so no bomb: just flee."""
    b = _strike_belief(_base_prior(grid_size=5), (1, 2), team_bombs=3)
    danger = DangerMap({(1, 2): 4}, b)
    p = build_planner(b, danger)
    assert survive(b, danger, p, StrategyParams()) == 0     # FORWARD, no PLACE_BOMB


def test_survive_avoids_dead_end_pocket():
    """Four equidistant safe cells; (0,3) is a walled-off pocket. survive's
    openness bias must pick an open cell, never the pocket (BACKWARD heads
    to the pocket from (3,3) facing RIGHT)."""
    prior = _base_prior(grid_size=7, wall_between={
        frozenset({(0, 3), (0, 2)}): False,
        frozenset({(0, 3), (0, 4)}): False,
    })
    b = _strike_belief(prior, (3, 3), team_bombs=3)
    danger = DangerMap({(3, 3): 4}, b)
    p = build_planner(b, danger)
    action = survive(b, danger, p, StrategyParams())
    assert action in (0, 2, 3)        # an open cell; never BACKWARD(1) into the pocket


# --- survive Tier 2: least-bad fallback, no STAY ---

def test_survive_tier2_moves_from_double_to_single():
    """Whole 5x5 grid is dangerous; row y=0 is single-damage, the rest double.
    From (2,1) facing UP, survive steps FORWARD onto the single-damage (2,0)."""
    b = _strike_belief(_base_prior(grid_size=5), (2, 1), team_bombs=3)
    b.facing = 3                                  # facing UP
    danger = DangerMap({(2, 2): 2, (2, 3): 2}, b)
    p = build_planner(b, danger)
    assert survive(b, danger, p, StrategyParams()) == 0     # FORWARD to lower overlap


def test_survive_tier2_yields_when_least_bad():
    """Whole grid uniformly single-damage; the agent is already on a least-bad
    cell -> survive yields (None) so the objective layers drive. Never STAY."""
    b = _strike_belief(_base_prior(grid_size=5), (0, 0), team_bombs=3)
    danger = DangerMap({(2, 2): 2}, b)            # one central bomb covers all 5x5
    p = build_planner(b, danger)
    assert survive(b, danger, p, StrategyParams()) is None


# --- strike: two-scenario wall breach -------------------------------------

def _walled_base_prior():
    """7x7 prior: enemy base at (3,3) sealed behind one destructible wall at
    (2,3)|(3,3); its other three sides are indestructible. The only way to a
    tile that can hit the base is to bomb that destructible wall open."""
    return _base_prior(grid_size=7, enemy_bases=((3, 3),), wall_between={
        frozenset({(3, 3), (2, 3)}): True,    # destructible — the only breach point
        frozenset({(3, 3), (4, 3)}): False,
        frozenset({(3, 3), (3, 2)}): False,
        frozenset({(3, 3), (3, 4)}): False,
    })


def test_strike_breaches_a_wall_to_reach_a_base():
    """With spare bombs and a base reachable only by breaching, strike drops a
    bomb now — scenario B reaches a hit-tile, scenario A never does."""
    b = _strike_belief(_walled_base_prior(), (1, 3), team_bombs=3)
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert strike(b, danger, p, StrategyParams()) == 5            # PLACE_BOMB


def test_strike_does_not_breach_with_only_one_bomb():
    """team_bombs below breach_min_bombs (2) -> strike never spends the bomb on
    a breach; with no other option it yields."""
    b = _strike_belief(_walled_base_prior(), (1, 3), team_bombs=1)
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    # Only one bomb: no breach, and the walled base is otherwise unreachable,
    # so strike yields to the next cascade layer.
    assert strike(b, danger, p, StrategyParams()) is None


def test_strike_skips_breach_when_an_ally_bomb_already_opens_the_wall():
    """An ally bomb already opening the breach wall makes scenario A reach the
    base for free -> scenario B is not strictly faster -> no breach bomb."""
    b = _strike_belief(_walled_base_prior(), (1, 3), team_bombs=3)
    b.ally_bombs = {(2, 3): 2}                 # blast opens (2,3)|(3,3) at tick 2
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    # Scenario A already reaches the base via the ally-opened wall, so strike
    # navigates with a move instead of breaching.
    assert strike(b, danger, p, StrategyParams()) in (0, 1, 2, 3)


# --- effective HP ----------------------------------------------------------

def test_effective_hp_subtracts_in_flight_own_bombs():
    from scripted.layers import _effective_hp
    b = _strike_belief(_base_prior(), (3, 3))
    b.enemy_base_health = {(3, 3): 0.8}          # 80 HP observed
    b.own_bombs = [((3, 3), 3)]                  # one own bomb reaches it
    assert _effective_hp(b, (3, 3)) == 60.0      # 80 - 20


def test_effective_hp_counts_stacked_bombs():
    from scripted.layers import _effective_hp
    b = _strike_belief(_base_prior(), (3, 3))
    b.enemy_base_health = {(3, 3): 0.8}
    b.own_bombs = [((3, 3), 3), ((3, 3), 2)]     # two bombs stacked on one tile
    assert _effective_hp(b, (3, 3)) == 40.0      # 80 - 40 — both counted


def test_effective_hp_floors_at_zero():
    from scripted.layers import _effective_hp
    b = _strike_belief(_base_prior(), (3, 3))
    b.enemy_base_health = {(3, 3): 0.6}          # 60 HP
    b.own_bombs = [((3, 3), 3)] * 5              # 100 dmg in flight
    assert _effective_hp(b, (3, 3)) == 0.0       # floored, never negative


def test_base_doomed_true_exactly_when_effective_hp_zero():
    from scripted.layers import _base_doomed
    b = _strike_belief(_base_prior(), (3, 3))
    b.enemy_base_health = {(3, 3): 0.4}
    b.own_bombs = [((3, 3), 3)]                  # 20 dmg -> 20 HP left
    assert _base_doomed(b, (3, 3)) is False
    b.own_bombs = [((3, 3), 3), ((3, 3), 3)]     # 40 dmg -> 0 HP
    assert _base_doomed(b, (3, 3)) is True


# --- _target_base ----------------------------------------------------------

def test_target_base_prefers_the_more_damaged_base():
    from scripted.layers import _target_base
    prior = _base_prior(grid_size=7, enemy_bases=((1, 1), (5, 5)))
    b = _strike_belief(prior, (3, 3))
    b.enemy_base_health = {(1, 1): 1.0, (5, 5): 0.4}   # (5,5) crippled
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    base, eff_hp, bombs_needed = _target_base(b, p, StrategyParams())
    assert base == (5, 5)                # fewer bombs to kill wins
    assert eff_hp == 40.0
    assert bombs_needed == 2


def test_target_base_none_when_no_live_base():
    from scripted.layers import _target_base
    prior = _base_prior(grid_size=7, enemy_bases=((1, 1),))
    b = _strike_belief(prior, (3, 3))
    b.dead_bases = {(1, 1)}
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert _target_base(b, p, StrategyParams()) is None


def test_target_base_blend_breaks_ties_by_distance():
    from scripted.layers import _target_base
    # two equal-HP bases; the nearer one wins on the travel term.
    prior = _base_prior(grid_size=7, enemy_bases=((1, 3), (6, 3)))
    b = _strike_belief(prior, (2, 3))                  # next to (1,3)
    b.enemy_base_health = {(1, 3): 1.0, (6, 3): 1.0}
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    base, _, _ = _target_base(b, p, StrategyParams())
    assert base == (1, 3)


def test_target_base_skips_a_doomed_base():
    from scripted.layers import _target_base
    prior = _base_prior(grid_size=7, enemy_bases=((1, 1), (5, 5)))
    b = _strike_belief(prior, (3, 3))
    b.enemy_base_health = {(1, 1): 0.2, (5, 5): 0.5}
    b.own_bombs = [((1, 1), 3)]            # one bomb finishes (1,1) -> doomed
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    base, _, _ = _target_base(b, p, StrategyParams())
    assert base == (5, 5)                  # the doomed base is skipped


# --- strike: two-phase soften / one-shot -----------------------------------

def test_strike_phase_a_softens_with_a_single_bomb():
    """A near-full base (effective_hp > soften_floor) is bombed with any bomb
    in hand — no commitment gate."""
    b = _strike_belief(_base_prior(), (3, 5), team_bombs=1)   # 2 tiles from base
    # no enemy_base_health record -> assumed full (100 HP) -> Phase A
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert strike(b, danger, p, StrategyParams()) == 5        # PLACE_BOMB


def test_strike_phase_b_gate_yields_when_under_stocked():
    """A crippled base (effective_hp <= soften_floor) is not struck without
    the full kill in hand — strike yields."""
    b = _strike_belief(_base_prior(), (3, 5), team_bombs=2)
    b.enemy_base_health = {(3, 3): 0.6}      # 60 HP -> needs 3 bombs
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert strike(b, danger, p, StrategyParams()) is None     # gated


def test_strike_phase_b_commits_with_the_full_kill():
    """Same crippled base, but the agent now holds the full 3 bombs."""
    b = _strike_belief(_base_prior(), (3, 5), team_bombs=3)
    b.enemy_base_health = {(3, 3): 0.6}
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert strike(b, danger, p, StrategyParams()) == 5        # PLACE_BOMB


def test_strike_phase_b_one_bomb_base_is_never_gated():
    """A base needing only one bomb is struck immediately."""
    b = _strike_belief(_base_prior(), (3, 5), team_bombs=1)
    b.enemy_base_health = {(3, 3): 0.2}      # 20 HP -> needs 1 bomb
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert strike(b, danger, p, StrategyParams()) == 5        # PLACE_BOMB


# --- sweep leash & default target ------------------------------------------

def test_sweep_leashes_collection_to_a_crippled_target_base():
    """With a Phase-B target base, sweep ignores collectibles outside the
    HP-scaled leash (bombs_needed + 1) of that base."""
    prior = _base_prior(grid_size=16, enemy_bases=((8, 8),))
    prior.collectibles = {(8, 9): 5.0, (1, 1): 5.0}   # near base; far corner
    b = _strike_belief(prior, (8, 11))                # within reach of (8,9)
    b.enemy_base_health = {(8, 8): 0.4}               # 40 HP -> bombs_needed 2 -> leash 3
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    action = sweep(b, danger, p, StrategyParams())
    # (1,1) is Chebyshev 7+ from the base -> outside leash 3 -> only (8,9) is
    # eligible, so sweep heads toward it (a real move, not None).
    assert action in (0, 1, 2, 3)
    # Confirm the far collectible alone yields nothing (it is leashed out).
    prior.collectibles = {(1, 1): 5.0}
    assert sweep(b, danger, p, StrategyParams()) is None


def test_sweep_unleashed_for_a_full_hp_target_base():
    """A Phase-A (full-HP) target imposes no leash — a far collectible is
    still pursued."""
    prior = _base_prior(grid_size=16, enemy_bases=((8, 8),))
    prior.collectibles = {(1, 1): 5.0}                # far from the base
    b = _strike_belief(prior, (2, 2))
    # no enemy_base_health -> full HP -> Phase A -> no leash
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert sweep(b, danger, p, StrategyParams()) in (0, 1, 2, 3)


def test_default_advances_toward_the_target_base():
    prior = _base_prior(grid_size=7, enemy_bases=((1, 1), (5, 5)))
    b = _strike_belief(prior, (3, 3))
    b.enemy_base_health = {(1, 1): 1.0, (5, 5): 0.4}  # (5,5) is the target
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert default(b, danger, p, StrategyParams()) in (0, 1, 2, 3)


# --- hunt & survive bomb discipline ----------------------------------------

def test_hunt_reserves_a_bomb_for_base_offense():
    """One enemy in range, 2 bombs: while a live base remains hunt holds fire
    (needs 3); with no live base it bombs (needs 2)."""
    from scripted.layers import hunt
    prior = _base_prior(grid_size=7, enemy_bases=((6, 6),))
    b = _strike_belief(prior, (3, 3), team_bombs=2)
    b.enemies = {(4, 3)}                          # one enemy, one tile ahead
    danger = DangerMap({}, b)
    p = build_planner(b, danger)
    assert hunt(b, danger, p, StrategyParams()) != 5      # base lives -> hold
    b.dead_bases = {(6, 6)}                               # endgame
    assert hunt(b, danger, p, StrategyParams()) == 5      # now bombs


def test_survive_skips_a_valueless_flee_drop_while_bases_live():
    """survive's opportunistic flee-drop is suppressed while a live base
    remains if the dropped bomb would reach neither a base nor an enemy."""
    prior = _base_prior(grid_size=5, enemy_bases=((4, 4),))
    b = _strike_belief(prior, (0, 0), team_bombs=3)       # corner, base far
    danger = DangerMap({(2, 0): 4}, b)                    # blast covers x0..3
    p = build_planner(b, danger)
    action = survive(b, danger, p, StrategyParams())
    assert action != 5                                    # no zero-value drop
