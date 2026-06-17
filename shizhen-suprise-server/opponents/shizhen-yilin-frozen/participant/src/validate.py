"""Final action-list simulator (PLAN B3 step 7, E): every action is checked
against a local simulation — gold ledger, occupancy including our own this-turn
placements, path cost vs movement budget on remembered terrain, range checks.
Invalid actions are silently dropped by the engine; here every drop is LOGGED,
because each one is a latent bug.
"""

from __future__ import annotations

import logging

from engine.actions import (
    AttackAction,
    BreakTreatyAction,
    ConstructBuildingAction,
    MoveAction,
    ProduceUnitAction,
    ProposeTreatyAction,
    RespondTreatyAction,
)
from engine.constants import BUILDING_STATS, TREATY_CUTOFF_TURN, UNIT_STATS

from world import WorldMemory, coord_of

log = logging.getLogger("validate")


def validate(world: WorldMemory, actions: list) -> list:
    grid = world.grid
    if grid is None:
        return []
    out: list = []
    gold = world.gold
    own_units = {u["id"]: u for u in world.own_units}
    own_buildings = {b["id"]: b for b in world.own_buildings}
    moved: set[str] = set()
    attacked: set[str] = set()
    claimed: set = set()  # endpoints + build tiles + spawn tiles this turn
    costs = world.move_costs()

    def reject(action, why: str) -> None:
        log.info("turn %s: dropped %s (%s)", world.turn, action, why)

    for a in actions:
        if isinstance(a, MoveAction):
            u = own_units.get(a.unit_id)
            if u is None or a.unit_id in moved:
                reject(a, "unknown unit or duplicate move")
                continue
            if len(a.path) < 2 or a.path[0] != coord_of(u):
                reject(a, "bad path origin/length")
                continue
            budget = u.get("movement_range", 0)
            if len(a.path) - 1 > budget:
                reject(a, "too many steps")
                continue
            if sum(costs.get(grid.wrap(s), 2) for s in a.path[1:]) > budget:
                reject(a, "path cost over budget")
                continue
            dest = grid.wrap(a.path[-1])
            occ = world.occupied.get(dest)
            if (occ is not None and occ["id"] != a.unit_id) or dest in claimed:
                reject(a, "destination occupied/claimed")
                continue
            moved.add(a.unit_id)
            claimed.add(dest)
            out.append(a)

        elif isinstance(a, AttackAction):
            u = own_units.get(a.unit_id)
            if u is None or a.unit_id in attacked:
                reject(a, "unknown unit or duplicate attack")
                continue
            d = grid.distance(coord_of(u), a.target)
            if d == 0 or d > u.get("attack_range", 0):
                reject(a, "target out of range")
                continue
            occ = world.occupied.get(grid.wrap(a.target))
            if occ is not None and (
                occ.get("owner_id") == world.player_id
                or world.at_peace_with(occ.get("owner_id"))
            ):
                reject(a, "friendly/allied on target tile")
                continue
            attacked.add(a.unit_id)
            out.append(a)

        elif isinstance(a, ConstructBuildingAction):
            stats = BUILDING_STATS.get(a.building_type)
            if stats is None or gold < stats.gold_cost:
                reject(a, "unknown type or unaffordable")
                continue
            c = grid.wrap(a.coord)
            occ = world.occupied.get(c)
            # own units may vacate-and-build (engine: moves resolve first; a
            # failed vacate makes the build a no-op with NO gold spent)
            occupied_hard = occ is not None and not (
                occ.get("owner_id") == world.player_id
                and occ["type"] not in BUILDING_STATS
            )
            if occupied_hard or c in claimed:
                reject(a, "tile occupied/claimed")
                continue
            if a.building_type == "Base":
                if c not in world.visible:
                    reject(a, "Base target not visible")
                    continue
            elif not any(
                b.get("is_complete") and grid.distance(coord_of(b), c) <= 1
                for b in world.own_buildings
            ):
                reject(a, "no completed anchor adjacent")
                continue
            gold -= stats.gold_cost
            claimed.add(c)
            out.append(a)

        elif isinstance(a, ProduceUnitAction):
            b = own_buildings.get(a.building_id)
            stats = UNIT_STATS.get(a.unit_type)
            if b is None or stats is None or not b.get("is_complete"):
                reject(a, "bad building or unit type")
                continue
            if a.unit_type not in BUILDING_STATS[b["type"]].producible_unit_types:
                reject(a, "building cannot produce this type")
                continue
            if gold < stats.gold_cost:
                reject(a, "unaffordable")
                continue
            if grid.distance(coord_of(b), a.target) > 1:
                reject(a, "spawn target not adjacent")
                continue
            gold -= stats.gold_cost
            claimed.add(grid.wrap(a.target))
            out.append(a)

        elif isinstance(
            a, (ProposeTreatyAction, RespondTreatyAction, BreakTreatyAction)
        ):
            if world.turn >= TREATY_CUTOFF_TURN:
                reject(a, "past treaty cutoff")
                continue
            out.append(a)

        else:
            out.append(a)  # HoldAction etc.

    return out
