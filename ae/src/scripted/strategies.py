"""Pluggable scripted strategies: a Strategy = (layer sequence, params).

See docs/superpowers/specs/2026-05-16-ae-scripted-strategies-design.md.
"""
from dataclasses import dataclass

from scripted.layers import camp, default, forage, hold, hunt, strike, survive, sweep


@dataclass(frozen=True)
class StrategyParams:
    """All strategy-tunable knobs. Env-physics constants stay in their modules."""

    sweep_base_gradient: float = 0.5   # weight of the drift-toward-enemy-base term
    forage_requires_endgame: bool = True   # forage self-disables while a base lives
    camp_leash: int | None = None      # camper territory radius; None => no leash
    hunt_max_route: float = 6.0        # max route cost hunt travels toward a bomb tile
    openness_radius: int = 4           # BFS cap for the survive dead-end openness score
    openness_weight: float = 1.5       # weight of openness vs distance in survive Tier 1
    bomb_drop_min: int = 2             # min team_bombs to drop a bomb while fleeing
    bomb_drop_buffer: int = 1          # tick cushion that must remain after the place tick
    breach_min_bombs: int = 2          # min team_bombs for strike to breach a wall
    target_travel_weight: float = 0.05  # blended-score weight on arrival ticks
    soften_floor: float = 60.0          # effective-HP boundary: soften vs one-shot


@dataclass(frozen=True)
class Strategy:
    """A named cascade composition."""

    name: str
    layers: tuple
    params: StrategyParams


_DEFAULT = StrategyParams()

STRATEGIES = {
    "balanced": Strategy(
        "balanced", (survive, hunt, strike, forage, sweep, default), _DEFAULT),
    "balanced_extreme": Strategy(
        "balanced_extreme", (hunt, strike, survive, forage, sweep, default),
        _DEFAULT),
    "base_rusher": Strategy(
        "base_rusher", (survive, strike, default), _DEFAULT),
    "base_rusher_extreme": Strategy(
        "base_rusher_extreme", (strike, survive, default), _DEFAULT),
    "collector": Strategy(
        "collector", (survive, forage, sweep, default),
        StrategyParams(forage_requires_endgame=False)),
    "camper": Strategy(
        "camper", (survive, camp, forage, sweep, hold),
        StrategyParams(camp_leash=4, forage_requires_endgame=False)),
}
