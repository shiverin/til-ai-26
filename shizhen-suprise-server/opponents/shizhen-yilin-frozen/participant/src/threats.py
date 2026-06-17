"""Threat assessment: defense trigger levels per base (PLAN C).

L0 (peace): garrison only
L1: hostile unit within 8 of a base — reinforce garrison
L2: hostile within 4, or a base has taken damage — all gold to defense
"""

from __future__ import annotations

from engine.hex_grid import HexCoord

from world import WorldMemory, coord_of

L0, L1, L2 = 0, 1, 2
L1_RADIUS = 8
L2_RADIUS = 4
WAR_PREP_LEAD = 20  # turns before the treaty void to re-arm (PLAN A6);
# 15 proved too late once production throughput is the bottleneck, not gold


class ThreatReport:
    def __init__(self) -> None:
        self.level = L0
        self.base_levels: dict[str, int] = {}  # base entity id -> level
        self.hostiles: list[dict] = []  # hostile unit records, nearest-first
        self.threatened_base: dict | None = None  # most-threatened base entity
        self.war_prep = False


def war_prep_turn(world: WorldMemory) -> int:
    from engine.constants import TREATY_CUTOFF_TURN

    return min(TREATY_CUTOFF_TURN, world.max_turns) - WAR_PREP_LEAD


def assess(world: WorldMemory) -> ThreatReport:
    rep = ThreatReport()
    rep.war_prep = world.turn >= war_prep_turn(world)
    # from war-prep on, every "ally" is a future enemy: their units count as
    # threats for DEFENSE POSTURE (attacks on partners stay forbidden elsewhere)
    rep.hostiles = world.hostile_units(include_partners=rep.war_prep)
    if world.grid is None:
        return rep

    # Mass-based triggers: with 19 turtling neighbours, a lone scout wandering
    # past is not an attack — panic-producing infantry over every passer-by
    # floods the home cluster and starves the economy (seen in self-play).
    worst_d = 10**9
    for base in world.bases:
        bc = coord_of(base)
        level = L0
        if base.get("hp", 0) < base.get("max_hp", 1):
            level = L2
        n4 = n8 = 0
        ranged_close = False
        for h in rep.hostiles:
            d = world.grid.distance(bc, coord_of(h))
            if d <= L2_RADIUS:
                n4 += 1
                if h.get("attack_range", 0) >= 2:
                    ranged_close = True
            if d <= L1_RADIUS:
                n8 += 1
            if d < worst_d:
                worst_d = d
                rep.threatened_base = base
        if n4 >= 2 or ranged_close:
            level = L2
        elif n4 >= 1 or n8 >= 3:
            level = max(level, L1)
        rep.base_levels[base["id"]] = level
        rep.level = max(rep.level, level)

    if rep.threatened_base is None and world.bases:
        rep.threatened_base = world.bases[0]

    # nearest-first ordering relative to the most relevant base
    if rep.threatened_base is not None and rep.hostiles:
        anchor: HexCoord = coord_of(rep.threatened_base)
        rep.hostiles.sort(key=lambda h: world.grid.distance(anchor, coord_of(h)))
    return rep
