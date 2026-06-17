"""Focus-fire allocator (PLAN B3 step 4, A5).

Simultaneous damage pool + no overkill prevention → assign exactly-lethal damage
per target in priority order, then spread surplus attackers onto the next-best
target in their range. Artillery has a friendly-splash veto (splash ignores
treaties AND hits the attacker's own ring-1 tile on adjacent shots).

Targets: anything visible this turn, plus remembered enemy BUILDINGS (static,
and attacks need no vision — verified in turn_processor). Treaty partners are
skipped entirely: attacking them is an invalid no-op.
"""

from __future__ import annotations

from engine.actions import AttackAction
from engine.constants import (
    ARTILLERY_SPLASH_RADIUS,
    BUILDING_STATS,
    ELEVATION_ATTACK_BONUS,
)

from threats import ThreatReport
from world import WorldMemory, coord_of

BOMBER_BUILDING_MULT = 4.0  # engine: Bomber.BUILDING_DAMAGE_MULTIPLIER
SCOUT_ALERT_RADIUS = 5  # an enemy Scout this close to a base has seen us


def plan_attacks(
    world: WorldMemory, threat: ThreatReport
) -> tuple[list[AttackAction], dict[str, object]]:
    """Returns (attack actions, unit_id -> target coord for units that fired)."""
    grid = world.grid
    if grid is None:
        return [], {}

    attackers = [u for u in world.own_units if u.get("attack_range", 0) > 0]
    if not attackers:
        return [], {}

    targets = _gather_targets(world)
    if not targets:
        return [], {}

    # splash veto: never splash own units or any treaty partner; own healthy
    # BUILDINGS may take chip splash (30 < the 200/turn a sieging unit deals) —
    # otherwise artillery can never shell attackers adjacent to our own walls
    no_splash = {coord_of(u) for u in world.own_units}
    no_splash |= {
        coord_of(b) for b in world.own_buildings if b.get("hp", 0) <= 150
    }
    no_splash |= {
        coord_of(e)
        for e in world.visible_enemies
        if world.at_peace_with(e.get("owner_id"))
    }

    # damage each attacker can deal to each target in range
    in_range: dict[str, list[dict]] = {t["id"]: [] for t in targets}
    dmg: dict[tuple[str, str], int] = {}
    for u in attackers:
        uc = coord_of(u)
        power = u.get("attack_power", 0)
        if world.terrain.get(uc) == "elevated":
            power = int(power * ELEVATION_ATTACK_BONUS)
        for t in targets:
            tc = coord_of(t)
            d = grid.distance(uc, tc)
            if d == 0 or d > u.get("attack_range", 0):
                continue
            if u["type"] == "Artillery" and _splash_unsafe(grid, tc, no_splash):
                continue
            amount = power
            if u["type"] == "Bomber" and t["type"] in BUILDING_STATS:
                amount = int(amount * BOMBER_BUILDING_MULT)
            in_range[t["id"]].append(u)
            dmg[(u["id"], t["id"])] = amount

    targets.sort(key=lambda t: _priority(world, t))
    assigned: dict[str, dict] = {}  # unit_id -> target entity

    # pass 1: exact-lethal allocation in priority order
    for t in targets:
        avail = [u for u in in_range[t["id"]] if u["id"] not in assigned]
        if not avail:
            continue
        pool = sum(dmg[(u["id"], t["id"])] for u in avail)
        if pool < t.get("hp", 0):
            continue  # can't kill this turn — leave attackers for later targets
        avail.sort(key=lambda u: dmg[(u["id"], t["id"])], reverse=True)
        need = t.get("hp", 0)
        for u in avail:
            if need <= 0:
                break
            assigned[u["id"]] = t
            need -= dmg[(u["id"], t["id"])]

    # pass 2: surplus attackers chip the highest-priority target in their range
    for u in attackers:
        if u["id"] in assigned:
            continue
        best = None
        for t in targets:
            if u in in_range[t["id"]]:
                best = t
                break
        if best is not None:
            assigned[u["id"]] = best

    actions: list[AttackAction] = []
    engaged: dict[str, object] = {}
    for uid, t in assigned.items():
        tc = coord_of(t)
        actions.append(AttackAction(unit_id=uid, target=tc))
        engaged[uid] = tc
    return actions, engaged


def _gather_targets(world: WorldMemory) -> list[dict]:
    targets: list[dict] = []
    seen: set[str] = set()
    for e in world.visible_enemies:
        if world.at_peace_with(e.get("owner_id")):
            continue
        targets.append(e)
        seen.add(e["id"])
    # remembered enemy buildings are static — blind fire is engine-legal
    for rec in world.enemy_buildings.values():
        if rec["id"] in seen or world.at_peace_with(rec.get("owner_id")):
            continue
        targets.append(rec)
    return targets


def _splash_unsafe(grid, target_coord, no_splash: set) -> bool:
    return any(
        c in no_splash for c in grid.ring(target_coord, ARTILLERY_SPLASH_RADIUS)
    )


def _priority(world: WorldMemory, t: dict) -> float:
    """Lower = shoot first. Artillery > Bomber > unit at our walls > Scout that
    has seen home > other units > buildings (inert last)."""
    grid = world.grid
    tc = coord_of(t)
    near_base = min(
        (grid.distance(coord_of(b), tc) for b in world.bases), default=99
    )
    tt = t["type"]
    if tt in BUILDING_STATS:
        if t.get("owner_id") in world.eliminated:
            return 90 + near_base / 100  # inert obstacle — only if nothing else
        if tt == "Base":
            return 50 + near_base / 100
        return 60 + near_base / 100
    if tt == "Artillery":
        return 0 + near_base / 100
    if tt == "Bomber":
        return 1 + near_base / 100
    near_buildings = min(
        (grid.distance(coord_of(b), tc) for b in world.own_buildings), default=99
    )
    if near_buildings <= 2:
        return 2 + near_buildings / 100
    if tt == "Scout" and near_base <= SCOUT_ALERT_RADIUS:
        return 3 + near_base / 100
    return 10 + near_base / 10
