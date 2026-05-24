"""Pluggable scripted strategies: a Strategy = (layer sequence, params).

See docs/superpowers/specs/2026-05-16-ae-scripted-strategies-design.md.
"""
from dataclasses import dataclass

from scripted.layers import (
    camp, default, forage_chain, hold, hunt, strike, survive, sweep,
)
from scripted.adaptive_layers import defend_intercept, forage_loop, rush_roi, trap
from scripted.gates import body_block_resolve, scripted_opening


@dataclass(frozen=True)
class StrategyParams:
    """All strategy-tunable knobs. Env-physics constants stay in their modules."""

    sweep_base_gradient: float = 0.5   # weight of the drift-toward-enemy-base term
    forage_requires_endgame: bool = True   # forage self-disables while a base lives
    camp_leash: int | None = None      # camper territory radius; None => no leash
    hunt_max_route: float = 6.0        # max route cost hunt travels toward a bomb tile
    openness_radius: int = 4           # BFS cap for the survive dead-end openness score
    openness_weight: float = 1.5       # weight of openness vs distance in survive Tier 1
    bomb_drop_min: int = 5             # min team_bombs to drop a bomb while fleeing
    bomb_drop_buffer: int = 1          # tick cushion that must remain after the place tick
    breach_min_bombs: int = 4          # min team_bombs for strike to breach a wall
    target_travel_weight: float = 0.05  # blended-score weight on arrival ticks
    soften_floor: float = 60.0          # effective-HP boundary: soften vs one-shot
    loop_commit_ticks: int = 20        # min ticks on a forage loop before a switch
    switch_factor: float = 0.6         # realised/estimate ratio below which to switch
    forage_yield_window: int = 30      # trailing-window length for realised forage yield
    roi_gate_margin: float = 0.15      # ROI hysteresis margin (target switch / forage gate)
    vulture_hp_boost: float = 2.0      # ROI multiplier weight for a base's missing HP
    defend_radius: int = 4             # Chebyshev radius around our base that triggers defend
    trap_enabled: bool = True          # the trap layer self-disables when False
    hunt_bomb_floor: int = 6           # while any enemy base lives, hunt holds fire below this
                                       # bomb count; with all bases dead the floor is ignored
    stuck_trigger_ticks: int = 2       # consecutive failed-move ticks before we declare stuck
    stuck_blacklist_ttl: int = 10      # ticks a blacklisted tile remains a soft obstacle
    centre_value_weight: float = -0.4   # multiplicative centre-bias on collectible value
                                       # (0.0 disables; tile_value *= 1 + w * centre_prox)
    contested_value_factor: float = 0.2   # multiplicative deflation on a collectible's
                                          # value when a visible enemy can reach it before
                                          # us (1.0 disables; 0.0 hard-ignores the tile)


@dataclass(frozen=True)
class Strategy:
    """A named cascade composition, optionally wrapped with post-decision gates.

    Gates run in `decide.act` after the cascade picks an action; each gate may
    return an override int or None to pass through. See `scripted.gates`.
    """

    name: str
    layers: tuple
    params: StrategyParams
    gates: tuple = ()


_DEFAULT = StrategyParams()

STRATEGIES = {
    "balanced": Strategy(
        "balanced", (survive, hunt, strike, forage_chain, sweep, default), 
        _DEFAULT,
        gates=(body_block_resolve, scripted_opening),
    ),
    "balanced_extreme": Strategy(
        "balanced_extreme",
        (hunt, strike, survive, forage_loop, sweep, default),
        _DEFAULT,
        gates=(body_block_resolve,)),
    "base_rusher": Strategy(
        "base_rusher", (survive, strike, default), _DEFAULT),
    "base_rusher_extreme": Strategy(
        "base_rusher_extreme", (strike, survive, default), _DEFAULT),
    "collector": Strategy(
        "collector", (survive, forage_chain, sweep, default),
        StrategyParams(forage_requires_endgame=False)),
    "camper": Strategy(
        "camper", (survive, camp, forage_chain, sweep, hold),
        StrategyParams(camp_leash=4, forage_requires_endgame=False)),
    "forager": Strategy(
        "forager", (survive, forage_loop, sweep, default), _DEFAULT),
    "lean_rush": Strategy(
        "lean_rush", (survive, hunt, rush_roi, sweep, default), _DEFAULT),
    "defender": Strategy(
        "defender", (survive, defend_intercept, forage_loop, sweep, hold),
        _DEFAULT),
    "adaptive": Strategy(
        "adaptive",
        (survive, rush_roi, trap, forage_loop, sweep, default),
        _DEFAULT),
    "balanced_extreme_opening": Strategy(
        "balanced_extreme_opening",
        (hunt, strike, survive, forage_chain, sweep, default),
        _DEFAULT,
        gates=(body_block_resolve, scripted_opening),
    ),
    "glass_cannon": Strategy(
        "glass_cannon", (strike, default), _DEFAULT),
    "pacifist": Strategy(
        "pacifist", (survive, forage_chain, sweep, hold),
        StrategyParams(forage_requires_endgame=False)),
    "hunter_killer": Strategy(
        "hunter_killer", (hunt, survive, default), _DEFAULT),
}
