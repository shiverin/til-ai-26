"""Enemy-proximity ('agent bias') devaluation for the forage layers.

A collectible inside a VISIBLE enemy's bomb-blast footprint (LOS + Chebyshev
BLAST_RADIUS) is devalued by `enemy_avoid_factor ** (BLAST_RADIUS + 1 - cheb)`,
so closer-to-enemy tiles are penalised more and the penalty compounds across
enemies. factor == 1.0 disables it. Walls (no LOS) and frozen enemies exempt a
cell.
"""
from scripted.belief import Belief
from scripted.blast import BLAST_RADIUS

F = 0.75


class _Prior:
    def __init__(self, grid_size, wall_between):
        self.grid_size = grid_size
        self.wall_between = wall_between
        self.collectibles = {}


def _belief(grid_size=8, wall_between=None, enemies=(), frozen=()):
    b = Belief()
    b.prior = _Prior(grid_size, wall_between or {})
    b.destroyed_walls = set()
    b.enemies = set(enemies)
    b.frozen_enemies = set(frozen)
    return b


def _penalty(b, cell, factor=F):
    from scripted.layers import _enemy_threat_penalty
    return _enemy_threat_penalty(b, cell, factor)


def test_adjacent_to_enemy_is_penalised_most():
    b = _belief(enemies=[(2, 2)])
    # Chebyshev 1, clear LOS -> 0.75 ** (2 + 1 - 1) = 0.75**2
    assert _penalty(b, (2, 3)) == F ** 2


def test_two_tiles_from_enemy_penalised_less():
    b = _belief(enemies=[(2, 2)])
    # Chebyshev 2, clear LOS -> 0.75 ** (2 + 1 - 2) = 0.75**1
    assert _penalty(b, (2, 4)) == F ** 1


def test_outside_blast_radius_no_penalty():
    b = _belief(enemies=[(2, 2)])
    # Chebyshev 3 -> outside BLAST_RADIUS -> no devaluation
    assert _penalty(b, (2, 5)) == 1.0


def test_factor_one_disables():
    b = _belief(enemies=[(2, 2)])
    assert _penalty(b, (2, 3), factor=1.0) == 1.0


def test_wall_blocks_los_exempts_cell():
    # Wall between (2,3) and (2,4) blocks LOS from enemy at (2,2) to (2,4).
    b = _belief(enemies=[(2, 2)],
                wall_between={frozenset({(2, 3), (2, 4)}): False})
    assert _penalty(b, (2, 4)) == 1.0


def test_frozen_enemy_does_not_threaten():
    b = _belief(enemies=[(2, 2)], frozen=[(2, 2)])
    assert _penalty(b, (2, 3)) == 1.0


def test_two_enemies_compound():
    b = _belief(enemies=[(2, 2), (2, 4)])
    # cell (2,3) is Chebyshev 1 from BOTH enemies, clear LOS -> (0.75**2)**2
    assert _penalty(b, (2, 3)) == (F ** 2) ** 2


def test_blast_radius_is_two():
    # guards the formula's reliance on the env blast radius
    assert BLAST_RADIUS == 2


# --- integration: the penalty must actually steer sweep's target ---------- #

class _SweepPrior:
    def __init__(self):
        self.grid_size = 11
        self.wall_between = {}
        self.enemy_bases = []
        self.our_base = (0, 0)
        self.resource_cells = None
        # A is closer to the agent (wins on raw rate); B is the safe option.
        self.collectibles = {(3, 2): 1.0, (7, 4): 1.0}


def _sweep_belief():
    b = Belief()
    b.prior = _SweepPrior()
    b.destroyed_walls = set()
    b.collected = set()
    b.location = (5, 0)
    b.facing = 0
    b.enemy_bombs = {}
    b.own_bombs = []
    b.enemies = {(3, 1)}          # threatens A=(3,2) (Chebyshev 1, clear LOS)
    b.frozen_enemies = set()
    b.dead_bases = set()
    b.enemy_base_health = {}
    return b


def _sweep_action(factor):
    from scripted.danger import DangerMap
    from scripted.layers import sweep
    from scripted.pathfind import build_planner
    from scripted.strategies import StrategyParams
    b = _sweep_belief()
    danger = DangerMap({}, b)
    planner = build_planner(b, danger)
    params = StrategyParams(centre_value_weight=0.0, enemy_avoid_factor=factor)
    action = sweep(b, danger, planner, params)
    return action, planner


def test_sweep_targets_near_enemy_tile_when_bias_disabled():
    action, planner = _sweep_action(factor=1.0)
    assert action == planner.first_action((3, 2))   # closer A wins on raw rate


def test_sweep_avoids_near_enemy_tile_when_bias_enabled():
    action, planner = _sweep_action(factor=0.75)
    assert action == planner.first_action((7, 4))   # safe B wins after penalty


# --- strategy wiring: only balanced_extreme_opening changes --------------- #

def test_balanced_extreme_opening_enables_bias_and_disables_centre():
    from scripted.strategies import STRATEGIES
    p = STRATEGIES["balanced_extreme_opening"].params
    assert p.enemy_avoid_factor == 0.75
    assert p.centre_value_weight == 0.0


def test_other_strategies_keep_defaults():
    from scripted.strategies import STRATEGIES, StrategyParams
    d = StrategyParams()
    for name in ("balanced", "collector", "forager", "adaptive"):
        p = STRATEGIES[name].params
        assert p.enemy_avoid_factor == d.enemy_avoid_factor      # 1.0 (off)
        assert p.centre_value_weight == d.centre_value_weight    # -0.4
