"""Economy (PLAN B3 step 6, C-Economy): threat-responsive production first, then
compound-interest expansion — Mine > rich Base > backup Base > Factory/Airbase.

Spawn-or-lose-gold semantics: every produce order targets a free neighbour and is
ledgered so a building is never queued past its free-neighbour capacity.
"""

from __future__ import annotations

from engine.actions import ConstructBuildingAction, ProduceUnitAction
from engine.constants import BUILDING_STATS, UNIT_STATS
from engine.hex_grid import HexCoord

from threats import L1, L2, ThreatReport
from world import WorldMemory, coord_of

DEFENSE_RESERVE = 150
RESERVE_FROM_TURN = 30
MINES_PER_BASE = 4  # ring bases with mines over time (income + melee wall)
MINE_PAYBACK_TURNS = 12  # stop building mines that can't repay by game end
EXPANSION_FROM_TURN = 12
BASE_SPACING_TURNS = 40
FACTORY_FROM_TURN = 30
AIRBASE_BUILD_BY = 188  # under construction by ~turn 193 per plan; start earlier
GARRISON_PER_BASE = {0: 3, L1: 6, L2: 99}  # L2: spawn capacity is the only cap


def decide(world: WorldMemory, threat: ThreatReport, reserved: set[HexCoord]) -> list:
    grid = world.grid
    if grid is None:
        return []
    actions: list = []
    gold = world.gold
    level = threat.level
    reserve = (
        DEFENSE_RESERVE if world.turn >= RESERVE_FROM_TURN and level < L2 else 0
    )

    prod_buildings = {
        bt: [
            b
            for b in world.own_buildings
            if b["type"] == bt and b.get("is_complete")
        ]
        for bt in ("Barracks", "Factory", "Airbase")
    }

    def produce(unit_type: str, source_type: str) -> bool:
        nonlocal gold
        cost = UNIT_STATS[unit_type].gold_cost
        if gold < cost:
            return False
        for b in prod_buildings[source_type]:
            target = _spawn_tile(world, b, reserved)
            if target is None:
                continue
            actions.append(
                ProduceUnitAction(
                    building_id=b["id"], unit_type=unit_type, target=target
                )
            )
            world.production_ledger.append(
                {
                    "building_id": b["id"],
                    "unit_type": unit_type,
                    "due_turn": world.turn + UNIT_STATS[unit_type].build_turns,
                    "target": target,
                }
            )
            reserved.add(target)
            gold -= cost
            return True
        return False

    n_bases = max(1, len(world.bases))

    # ── units: defense first ──────────────────────────────────────────────────
    # war-prep (PLAN A6): the turn-200 void hits everyone at once — convert the
    # war chest into a standing army before it does
    inf_per_base = GARRISON_PER_BASE.get(level, 3)
    art_per_base = 2
    med_per_base = 1
    at_war = threat.war_prep or level >= L2
    if threat.war_prep:
        inf_per_base = max(inf_per_base, 8)
        art_per_base = 6
        med_per_base = 2
    # at war, ranged + air defense first: Fighter swarms killed us in testing —
    # only Fighters (and massed Artillery) can answer range-2/move-3 attackers
    if at_war:
        while world.our_unit_count("Fighter") < 2 * n_bases and produce(
            "Fighter", "Airbase"
        ):
            pass
        while world.our_unit_count("Artillery") < art_per_base * n_bases and produce(
            "Artillery", "Factory"
        ):
            pass
    inf_target = inf_per_base * n_bases
    while world.our_unit_count("Infantry") < inf_target and produce("Infantry", "Barracks"):
        pass
    if world.our_unit_count("Medic") < med_per_base * n_bases:
        produce("Medic", "Barracks")
    while world.our_unit_count("Artillery") < art_per_base * n_bases and produce(
        "Artillery", "Factory"
    ):
        pass
    # war: drain the remaining chest into Tanks (mobile HP blocks)
    if at_war and gold > 1000:
        while world.our_unit_count("Tank") < 2 * n_bases and produce("Tank", "Factory"):
            pass

    # scouts: #1 immediately, #2 from turn 8 (A/B candidate per plan)
    scouts = world.our_unit_count("Scout")
    if scouts < 1 or (scouts < 2 and world.turn >= 8):
        produce("Scout", "Barracks")

    # post-airbase deterrence package: 2 Bombers, then 1 Fighter with surplus
    if world.our_unit_count("Bomber") < 2:
        produce("Bomber", "Airbase")
    elif world.our_unit_count("Fighter") < 2 and gold > 800:
        produce("Fighter", "Airbase")

    def afford(building_type: str) -> bool:
        return gold - reserve >= BUILDING_STATS[building_type].gold_cost

    def build(building_type: str, coord: HexCoord) -> None:
        nonlocal gold
        actions.append(
            ConstructBuildingAction(building_type=building_type, coord=coord)
        )
        reserved.add(coord)
        gold -= BUILDING_STATS[building_type].gold_cost

    # a Barracks when we have NONE is survival infrastructure, not expansion —
    # build it even under fire (losing the last one would end unit production).
    # Likewise rebuild throughput when our spawn rings have silted up.
    spawn_capacity = sum(
        len(
            [
                c
                for c in grid.neighbors(coord_of(b))
                if c not in world.occupied and c not in reserved
            ]
        )
        for b in world.own_buildings
        if b["type"] in ("Barracks", "Factory", "Airbase") and b.get("is_complete")
    )
    barracks_cap = 2 + world.building_count("Base")
    if (
        world.building_count("Barracks") == 0
        or (spawn_capacity < 2 and world.building_count("Barracks") < barracks_cap)
    ) and gold >= 100:
        spot = _building_spot(world, "Barracks", reserved)
        if spot:
            build("Barracks", spot)

    # ── other buildings (frozen at L2 — every coin goes to units) ─────────────
    if level >= L2:
        return actions

    # Mines: rich tiles first; cap scales with bases, profitable until ~T-12
    mine_cap = MINES_PER_BASE * max(1, world.building_count("Base"))
    mines_now = world.building_count("Mine")
    while (
        mines_now < mine_cap
        and afford("Mine")
        and world.turn <= world.max_turns - MINE_PAYBACK_TURNS
    ):
        spot = _building_spot(world, "Mine", reserved)
        if spot is None:
            break
        build("Mine", spot)
        mines_now += 1
        if mines_now >= 2 and world.turn < EXPANSION_FROM_TURN:
            break  # early game: two mines, then let gold pool for scouts/Barracks

    # expansion / redundant Bases (extra lives)
    total_bases = world.building_count("Base")
    want_base = (
        world.turn >= EXPANSION_FROM_TURN
        and world.turn - world.last_base_order_turn >= BASE_SPACING_TURNS // 2
        and (
            total_bases < 2
            or (total_bases < 4 and gold - reserve >= 600)
        )
        and afford("Base")
    )
    if want_base:
        spot = _base_spot(world, reserved)
        if spot is not None:
            build("Base", spot)
            world.last_base_order_turn = world.turn
            world.expansion_goal = None
        else:
            # nothing suitable in sight — pick a remembered quiet tile and let
            # a scout walk there to provide the build-time vision (B4)
            world.expansion_goal = _expansion_goal(world)
    else:
        world.expansion_goal = None

    # Factory for Artillery defense
    if (
        world.turn >= FACTORY_FROM_TURN
        and world.building_count("Factory") == 0
        and gold - reserve >= BUILDING_STATS["Factory"].gold_cost + 100
    ):
        spot = _building_spot(world, "Factory", reserved)
        if spot:
            build("Factory", spot)

    # second Barracks near an expansion base that has none
    if world.building_count("Barracks") >= 1 and afford("Barracks"):
        for base in world.bases:
            bc = coord_of(base)
            if any(
                b["type"] == "Barracks"
                and world.grid.distance(bc, coord_of(b)) <= 2
                for b in world.own_buildings
            ):
                continue
            spot = _building_spot(world, "Barracks", reserved, near=bc)
            if spot:
                build("Barracks", spot)
            break

    # Airbase: in time for the turn-200 treaty void, or earlier on big surplus;
    # a second one at war for Fighter (anti-air) throughput
    airbase_cap = 2 if (threat.war_prep and gold - reserve >= 1000) else 1
    if world.building_count("Airbase") < airbase_cap and (
        (world.turn >= AIRBASE_BUILD_BY and afford("Airbase"))
        or gold - reserve >= 1500
    ):
        spot = _building_spot(world, "Airbase", reserved)
        if spot:
            build("Airbase", spot)

    return actions


# ── placement helpers ──────────────────────────────────────────────────────────


def _free_for_build(world: WorldMemory, c: HexCoord, reserved) -> bool:
    """Free, or occupied by an OWN UNIT we can vacate this turn (A5.6: moves
    resolve before builds; if the vacate fails the build is a free no-op)."""
    if c in reserved or c not in world.visible:
        return False
    occ = world.occupied.get(c)
    if occ is None:
        return True
    return (
        occ.get("owner_id") == world.player_id
        and occ["type"] not in BUILDING_STATS
        and occ.get("movement_range", 0) >= 1
    )


def _boxes_production(world: WorldMemory, c: HexCoord) -> bool:
    """Never take the last free neighbour of a production building. Own-unit
    tiles count as free — units vacate; buildings are forever."""
    for b in world.own_buildings:
        if b["type"] not in ("Barracks", "Factory", "Airbase"):
            continue
        bc = coord_of(b)
        if world.grid.distance(bc, c) != 1:
            continue
        free = 0
        for n in world.grid.neighbors(bc):
            if n == c:
                continue
            occ = world.occupied.get(n)
            if occ is None or (
                occ.get("owner_id") == world.player_id
                and occ["type"] not in BUILDING_STATS
            ):
                free += 1
        if free <= world.pending_spawns_for(b["id"]):
            return True
    return False


def _building_spot(
    world: WorldMemory, building_type: str, reserved, near: HexCoord | None = None
) -> HexCoord | None:
    """A free tile adjacent to a completed own building. Mines prefer rich tiles;
    everything else avoids squatting on them."""
    grid = world.grid
    anchors = [b for b in world.own_buildings if b.get("is_complete")]
    if near is not None:
        anchors.sort(key=lambda b: grid.distance(coord_of(b), near))
    spawn_ring: set[HexCoord] = set()
    for b in world.own_buildings:
        if b["type"] in ("Barracks", "Factory", "Airbase"):
            spawn_ring.update(grid.neighbors(coord_of(b)))
    best, best_score = None, -(10**9)
    for b in anchors:
        for c in grid.neighbors(coord_of(b)):
            if not _free_for_build(world, c, reserved) or _boxes_production(world, c):
                continue
            rich = c in world.rich_tiles
            score = 0
            if building_type == "Mine":
                score += 100 if rich else 0
            else:
                score -= 50 if rich else 0
            if c in spawn_ring:
                score -= 60  # don't eat production spawn tiles
            if near is not None:
                score -= grid.distance(c, near)
            if score > best_score:
                best, best_score = c, score
    return best


def _base_spot(world: WorldMemory, reserved) -> HexCoord | None:
    """Best VISIBLE tile for a new Base (engine demands current-turn vision):
    rich ≫ quiet corner, away from our other bases and from enemy BASES. Other
    enemy buildings (mines etc.) only penalise — with 19 neighbours sprawling,
    a hard keep-out around every shed would exclude the entire map."""
    grid = world.grid
    own_bases = [coord_of(b) for b in world.own_buildings if b["type"] == "Base"]
    enemy_bases = [
        coord_of(r)
        for r in world.enemy_buildings.values()
        if r["type"] == "Base" and r.get("owner_id") not in world.eliminated
    ]
    enemy_other = [
        coord_of(r)
        for r in world.enemy_buildings.values()
        if r["type"] != "Base" and r.get("owner_id") not in world.eliminated
    ]
    best, best_score = None, -(10**9)
    for c in world.visible:
        if c in world.occupied or c in reserved:
            continue
        d_own = min((grid.distance(c, b) for b in own_bases), default=99)
        if d_own < 4:
            continue  # redundancy demands separation
        d_ebase = min((grid.distance(c, e) for e in enemy_bases), default=99)
        if d_ebase < 6:
            continue  # crowding a neighbour's home invites the wars we're avoiding
        d_eother = min((grid.distance(c, e) for e in enemy_other), default=99)
        score = (
            (200 if c in world.rich_tiles else 0)
            + min(d_ebase, 15)
            + min(d_eother, 6)
            - d_own
        )
        if score > best_score:
            best, best_score = c, score
    return best


def _expansion_goal(world: WorldMemory) -> HexCoord | None:
    """A remembered (not necessarily visible) tile worth founding a Base on."""
    grid = world.grid
    own_bases = [coord_of(b) for b in world.own_buildings if b["type"] == "Base"]
    enemy_bases = [
        coord_of(r)
        for r in world.enemy_buildings.values()
        if r["type"] == "Base" and r.get("owner_id") not in world.eliminated
    ]
    best, best_score = None, -(10**9)
    for c in world.terrain:
        if c in world.occupied:
            continue
        d_own = min((grid.distance(c, b) for b in own_bases), default=99)
        if d_own < 5:
            continue
        d_ebase = min((grid.distance(c, e) for e in enemy_bases), default=99)
        if d_ebase < 7:
            continue
        score = (150 if c in world.rich_tiles else 0) + min(d_ebase, 15) - d_own
        if score > best_score:
            best, best_score = c, score
    return best


def _spawn_tile(world: WorldMemory, building: dict, reserved) -> HexCoord | None:
    bc = coord_of(building)
    free = [
        c
        for c in world.grid.neighbors(bc)
        if c not in world.occupied and c not in reserved
    ]
    # spawn-or-lose-gold: leave room for everything already queued here
    if len(free) <= world.pending_spawns_for(building["id"]):
        return None
    return free[0]
