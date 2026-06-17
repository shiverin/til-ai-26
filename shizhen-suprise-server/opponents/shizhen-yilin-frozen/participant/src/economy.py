"""Economy (PLAN B3 step 6, C-Economy): threat-responsive production first, then
compound-interest expansion — Mine > rich Base > backup Base > Factory/Airbase.

Spawn-or-lose-gold semantics: every produce order targets a free neighbour and is
ledgered so a building is never queued past its free-neighbour capacity.
"""

from __future__ import annotations

import logging

from engine.actions import ConstructBuildingAction, ProduceUnitAction
from engine.constants import BUILDING_STATS, UNIT_STATS
from engine.hex_grid import HexCoord

from threats import L1, L2, ThreatReport
from world import WorldMemory, coord_of

log = logging.getLogger("economy")

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
    # wealth pressure: once the treasury outgrows every fixed sink, raise the
    # army/building targets so income converts into force instead of pooling
    # (seed-67 multi-agent run: 121k idle gold while being ground down at L2)
    surplus = max(0, gold - reserve)
    wealth_bonus = min(30, surplus // 1000)

    prod_buildings = {
        bt: [
            b
            for b in world.own_buildings
            if b["type"] == bt and b.get("is_complete")
        ]
        for bt in ("Barracks", "Factory", "Airbase")
    }

    at_war = threat.war_prep or level >= L2

    def produce(unit_type: str, source_type: str) -> bool:
        nonlocal gold
        cost = UNIT_STATS[unit_type].gold_cost
        if gold < cost:
            return False
        for b in prod_buildings[source_type]:
            # at war, throughput beats the safety buffer: every spawn tile
            # counts (cb31973a: capacity-starved at L2 with 100k banked)
            target = _spawn_tile(world, b, reserved, relax=at_war)
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
                    "ordered_turn": world.turn,
                }
            )
            reserved.add(target)
            gold -= cost
            return True
        if prod_buildings[source_type]:
            # cada2771 post-mortem: this exact stall (silted spawn rings) is how
            # 139k gold died to a Fighter swarm — keep it loud even at WARNING
            log.warning(
                "turn %s: want %s but no free spawn tile on any %s",
                world.turn,
                unit_type,
                source_type,
            )
        return False

    n_bases = max(1, len(world.bases))

    # ── units: defense first ──────────────────────────────────────────────────
    # war-prep (PLAN A6): the turn-200 void hits everyone at once — convert the
    # war chest into a standing army before it does
    inf_per_base = GARRISON_PER_BASE.get(level, 3)
    art_per_base = 2
    med_per_base = 1
    if threat.war_prep:
        # 1ff4e9f5: 70 Infantry melted to 17 against air blitzes for ~zero
        # value — at war Infantry is chaff; the gold belongs in the sky
        inf_per_base = 4
        art_per_base = 6
        med_per_base = 2
    # at war, ranged + air defense first: Fighter swarms killed us in testing —
    # only Fighters (and massed Artillery) can answer range-2/move-3 attackers.
    # Match the air race we can OBSERVE: opponents converted their whole chest
    # to flyers at the void and outnumbered us 78:39 in the air (1ff4e9f5)
    enemy_air = sum(
        1
        for r in world.enemy_units.values()
        if r["type"] in ("Fighter", "Bomber")
        and r.get("owner_id") not in world.eliminated
    )
    if at_war:
        fighter_target = max(2 * n_bases + wealth_bonus, enemy_air + enemy_air // 5)
        while world.our_unit_count("Fighter") < fighter_target and produce(
            "Fighter", "Airbase"
        ):
            pass
        while world.our_unit_count(
            "Artillery"
        ) < art_per_base * n_bases + wealth_bonus and produce("Artillery", "Factory"):
            pass
    inf_target = inf_per_base * n_bases
    while world.our_unit_count("Infantry") < inf_target and produce("Infantry", "Barracks"):
        pass
    if world.our_unit_count("Medic") < med_per_base * n_bases:
        produce("Medic", "Barracks")
    while world.our_unit_count(
        "Artillery"
    ) < art_per_base * n_bases + wealth_bonus and produce("Artillery", "Factory"):
        pass
    # war: drain the remaining chest into Tanks (mobile HP blocks)
    if at_war and gold > 1000:
        while world.our_unit_count("Tank") < 2 * n_bases + wealth_bonus and produce(
            "Tank", "Factory"
        ):
            pass

    # scouts: #1 immediately, #2 from turn 8 (A/B candidate per plan)
    scouts = world.our_unit_count("Scout")
    if scouts < 1 or (scouts < 2 and world.turn >= 8):
        produce("Scout", "Barracks")

    # post-airbase standing air force, growing with the treasury. Fighters
    # FIRST: they are the only answer to enemy Fighter swarms (move 3 / range 2
    # kites everything on the ground), and gemini's 16-Fighter sweep at the
    # turn-200 void is what killed every base in cada2771. Bombers after.
    fighter_floor = max(2, min(24, surplus // 1200))
    while world.our_unit_count("Fighter") < fighter_floor and produce(
        "Fighter", "Airbase"
    ):
        pass
    bomber_floor = max(2, min(8, surplus // 2500))
    while world.our_unit_count("Bomber") < bomber_floor and produce(
        "Bomber", "Airbase"
    ):
        pass

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
    # ...unless the treasury dwarfs unit throughput: then gold is NOT the
    # constraint, and refusing to rebuild lost Mines/Factory/Airbase under
    # pressure is how a 100k war chest dies with 3 Barracks (seed-67 FFA).
    if level >= L2 and gold < 1500:
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
    # at war, redundancy IS the win condition: losing 4 bases in 15 turns while
    # the expansion timer allowed 1 rebuild per 20 is how cb31973a went 6 bases
    # -> 1. Below 3 bases the spacing gate shrinks to a token anti-spam delay.
    rebuild_rush = at_war and total_bases < 3
    spacing = 3 if rebuild_rush else BASE_SPACING_TURNS // 2
    want_base = (
        world.turn >= EXPANSION_FROM_TURN
        and world.turn - world.last_base_order_turn >= spacing
        and (
            rebuild_rush
            or total_bases < 2
            or (total_bases < 4 and gold - reserve >= 600)
            or (total_bases < 6 and gold - reserve >= 5000)
            or (total_bases < 8 and gold - reserve >= 20000)
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

    # Heavy production must BREATHE: one Factory/Airbase with a silted ring
    # cannot convert a six-figure war chest into Artillery/Fighters (cada2771:
    # Factory at 0 free neighbours from turn 150 → death at 221 with 139k gold).
    # Add another whenever the existing rings are starved or wealth allows.
    def ring_free(bt: str) -> int:
        return sum(
            1
            for b in prod_buildings[bt]
            for c in grid.neighbors(coord_of(b))
            if c not in world.occupied and c not in reserved
        )

    # Factory for Artillery defense
    # caps scale with the treasury: f276a8bb ended with 205k unspent because 3
    # Factories/Airbases were the conversion ceiling, not gold
    factory_cap = min(
        7 if at_war else 5,
        1 + surplus // 4000 + (1 if at_war else 0),
    )
    n_factories = world.building_count("Factory")
    if (
        world.turn >= FACTORY_FROM_TURN
        and n_factories < factory_cap
        and (n_factories == 0 or ring_free("Factory") < 2 or surplus >= 4000)
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
    # more at wealth/war for Fighter (anti-air) throughput
    airbase_cap = min(
        7 if at_war else 5,
        1 + surplus // 4000 + (1 if at_war and surplus >= 1000 else 0),
    )
    n_airbases = world.building_count("Airbase")
    if (
        n_airbases < airbase_cap
        and (
            (world.turn >= AIRBASE_BUILD_BY and afford("Airbase"))
            or gold - reserve >= 1500
        )
        and (n_airbases == 0 or ring_free("Airbase") < 2 or surplus >= 4000)
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
            if not _free_for_build(world, c, reserved):
                continue
            if c in spawn_ring:
                continue  # spawn rings are sacred — a soft penalty here is how
                # the Factory ended up at 0 free neighbours (cada2771)
            rich = c in world.rich_tiles
            score = 0
            if building_type == "Mine":
                score += 100 if rich else 0
            else:
                score -= 50 if rich else 0
            if building_type in ("Barracks", "Factory", "Airbase"):
                # production wants elbow room: reward open rings of its own
                score += 8 * sum(
                    1
                    for n in grid.neighbors(c)
                    if n not in world.occupied and n not in reserved
                )
            if near is not None:
                score -= grid.distance(c, near)
            if score > best_score:
                best, best_score = c, score
    return best


GRAVEYARD_MEMORY_TURNS = 40
GRAVEYARD_RADIUS = 5
HOSTILE_KEEPOUT = 4


def _danger_zones(world: WorldMemory) -> tuple[list[HexCoord], list[HexCoord]]:
    """Tiles a new Base must avoid: recently seen hostile units (the pack that
    killed the last base is still hunting) and recent base graveyards (cb31973a:
    rebuilds 3 tiles from the previous kill died to the same Fighter/Bomber pack)."""
    hostiles = [
        coord_of(r)
        for r in world.enemy_units.values()
        if r.get("owner_id") not in world.eliminated
        and world.turn - r.get("last_seen", 0) <= 8
        and not world.at_peace_with(r.get("owner_id"))
    ]
    graves = [
        c
        for c, t in world.base_graveyard
        if world.turn - t <= GRAVEYARD_MEMORY_TURNS
    ]
    return hostiles, graves


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
    hostiles, graves = _danger_zones(world)
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
        d_host = min((grid.distance(c, h) for h in hostiles), default=99)
        if d_host < HOSTILE_KEEPOUT:
            continue  # founding under a hunting pack is a 300g donation
        d_grave = min((grid.distance(c, g) for g in graves), default=99)
        if d_grave < GRAVEYARD_RADIUS:
            continue  # the killer is still patrolling here
        d_eother = min((grid.distance(c, e) for e in enemy_other), default=99)
        score = (
            (200 if c in world.rich_tiles else 0)
            + min(d_ebase, 15)
            + min(d_eother, 6)
            + 2 * min(d_host, 10)
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
    hostiles, graves = _danger_zones(world)
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
        if min((grid.distance(c, h) for h in hostiles), default=99) < HOSTILE_KEEPOUT:
            continue
        if min((grid.distance(c, g) for g in graves), default=99) < GRAVEYARD_RADIUS:
            continue
        score = (150 if c in world.rich_tiles else 0) + min(d_ebase, 15) - d_own
        if score > best_score:
            best, best_score = c, score
    return best


def _spawn_tile(
    world: WorldMemory, building: dict, reserved, relax: bool = False
) -> HexCoord | None:
    """A free neighbour not already promised to a pending spawn (spawn-or-lose-
    gold: two orders landing on one tile silently burn the second unit)."""
    bc = coord_of(building)
    pending = {p["target"] for p in world.production_ledger}
    free = [
        c
        for c in world.grid.neighbors(bc)
        if c not in world.occupied and c not in reserved and c not in pending
    ]
    # demand a buffer tile beyond the target: completion is build_turns away,
    # and a ring that is full-minus-one at order time tends to be full at
    # completion (c2f5fc26: 8 Fighters silently lost to exactly this drift).
    # At war (relax) the buffer is waived: risking one unit to drift beats
    # leaving the war chest unconverted while bases fall (cb31973a).
    if relax:
        return free[0] if free else None
    return free[0] if len(free) >= 2 else None
