"""Deterministic competition agent.

The shipped template survives the local random baseline, but it leaves most of
its gold idle. This agent stays offline and deterministic while using the
important engine rules: multi-queue production, extra Bases on visible tiles,
attack+move in the same turn, and focused defensive targeting.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from agent_base import PlayerAgent
from engine.actions import (
    ActionPayload,
    AttackAction,
    BreakTreatyAction,
    ConstructBuildingAction,
    MoveAction,
    ProduceUnitAction,
    ProposeTreatyAction,
    RespondTreatyAction,
)
from engine.constants import BUILDING_STATS, TREATY_CUTOFF_TURN, UNIT_STATS
from engine.hex_grid import HexCoord, HexGrid


_PRODUCTION_BUILDINGS = ("Barracks", "Factory", "Airbase")
_RESOURCE_BUILDINGS = ("Base", "Mine")
_BUILD_LIMIT_PER_TURN = 6
_PRODUCE_LIMIT_PER_TURN = 14


@dataclass
class World:
    pid: str
    turn: int
    max_turns: int
    gold: int
    grid: HexGrid
    own_units: list[dict]
    own_buildings: list[dict]
    enemies: list[dict]
    occupied: set[HexCoord]
    visible: set[HexCoord]
    terrain: dict[HexCoord, str]
    peace_players: set[str]
    known_players: set[str]
    incoming_treaty_proposals: list[dict]


class AlgoAgent(PlayerAgent):
    def __init__(self) -> None:
        self._seen_terrain: dict[HexCoord, str] = {}
        self._last_seen_enemy: dict[str, tuple[HexCoord, int, str]] = {}
        self._proposed_peace: set[str] = set()

    async def decide(self, observation: dict) -> ActionPayload:
        world = self._parse_world(observation)
        actions: list = []
        reserved = set(world.occupied)
        planned_gold = world.gold

        actions.extend(self._diplomacy_actions(world))

        combat_actions, reserved_after_moves = self._combat_and_moves(world, reserved)
        actions.extend(combat_actions)
        reserved = reserved_after_moves

        build_actions, planned_gold, reserved = self._build_actions(
            world, planned_gold, reserved
        )
        actions.extend(build_actions)

        produce_actions, planned_gold, reserved = self._production_actions(
            world, planned_gold, reserved
        )
        actions.extend(produce_actions)

        return ActionPayload(
            player_id=world.pid, turn_number=world.turn, actions=actions
        )

    # ── world parsing ────────────────────────────────────────────────────────

    def _parse_world(self, observation: dict) -> World:
        pid = observation["player_id"]
        turn = int(observation.get("turn_number", 0))
        grid = HexGrid(
            int(observation.get("map_width", 35)),
            int(observation.get("map_height", 30)),
        )
        own_units: list[dict] = []
        own_buildings: list[dict] = []
        enemies: list[dict] = []
        occupied: set[HexCoord] = set()
        visible: set[HexCoord] = set()
        terrain: dict[HexCoord, str] = {}

        for tile in observation.get("visible_tiles", []):
            coord = grid.wrap(HexCoord(int(tile["q"]), int(tile["r"])))
            visible.add(coord)
            terrain_name = tile.get("terrain", "normal")
            terrain[coord] = terrain_name
            self._seen_terrain[coord] = terrain_name
            for entity in tile.get("entities", []):
                ecoord = grid.wrap(HexCoord(int(entity["q"]), int(entity["r"])))
                occupied.add(ecoord)
                entity["q"], entity["r"] = ecoord.q, ecoord.r
                owner = entity.get("owner_id")
                if owner == pid:
                    if entity.get("type") in BUILDING_STATS:
                        own_buildings.append(entity)
                    else:
                        own_units.append(entity)
                else:
                    enemies.append(entity)
                    self._last_seen_enemy[entity["id"]] = (
                        ecoord,
                        turn,
                        entity.get("type", ""),
                    )

        peace_players = {
            t.get("partner_id", "")
            for t in observation.get("treaties", [])
            if t.get("treaty_type") == "peace" and t.get("partner_id")
        }
        enemies = [e for e in enemies if e.get("owner_id") not in peace_players]

        return World(
            pid=pid,
            turn=turn,
            max_turns=int(observation.get("max_turns", 300)),
            gold=int(observation.get("resources", {}).get("gold", 0)),
            grid=grid,
            own_units=own_units,
            own_buildings=own_buildings,
            enemies=enemies,
            occupied=occupied,
            visible=visible,
            terrain=terrain,
            peace_players=peace_players,
            known_players=set(observation.get("known_players", [])),
            incoming_treaty_proposals=list(
                observation.get("incoming_treaty_proposals", [])
            ),
        )

    # ── diplomacy ────────────────────────────────────────────────────────────

    def _diplomacy_actions(self, world: World) -> list:
        if world.turn >= min(TREATY_CUTOFF_TURN, world.max_turns):
            return [
                BreakTreatyAction(partner_player_id=pid, treaty_type="peace")
                for pid in sorted(world.peace_players)
            ]

        actions: list = []
        for proposal in world.incoming_treaty_proposals:
            proposer = proposal.get("proposer_id")
            if proposer:
                actions.append(
                    RespondTreatyAction(
                        proposing_player_id=proposer,
                        treaty_type=proposal.get("treaty_type", "peace"),
                        accept=True,
                    )
                )

        for pid in sorted(world.known_players):
            if (
                pid != world.pid
                and pid not in world.peace_players
                and pid not in self._proposed_peace
            ):
                actions.append(ProposeTreatyAction(target_player_id=pid))
                self._proposed_peace.add(pid)
        return actions[:8]

    # ── building economy ─────────────────────────────────────────────────────

    def _build_actions(
        self, world: World, gold: int, reserved: set[HexCoord]
    ) -> tuple[list, int, set[HexCoord]]:
        actions: list = []
        complete = [b for b in world.own_buildings if b.get("is_complete", True)]
        bases = [b for b in world.own_buildings if b["type"] == "Base"]
        complete_bases = [b for b in bases if b.get("is_complete", True)]
        mines = [b for b in world.own_buildings if b["type"] == "Mine"]
        barracks = [b for b in world.own_buildings if b["type"] == "Barracks"]
        factories = [b for b in world.own_buildings if b["type"] == "Factory"]
        airbases = [b for b in world.own_buildings if b["type"] == "Airbase"]
        own_unit_count = len(world.own_units)
        threatened = self._base_threatened(world)

        def try_build(building_type: str, coord: HexCoord) -> bool:
            nonlocal gold
            if len(actions) >= _BUILD_LIMIT_PER_TURN:
                return False
            cost = BUILDING_STATS[building_type].gold_cost
            if gold < cost or coord in reserved:
                return False
            actions.append(
                ConstructBuildingAction(building_type=building_type, coord=coord)
            )
            gold -= cost
            reserved.add(coord)
            return True

        # Opening: spend the starting 500 immediately on production + economy.
        if world.turn <= 1 and complete:
            for building_type in ("Barracks", "Mine", "Mine"):
                spot = self._best_adjacent_build_tile(
                    world, complete, reserved, building_type
                )
                if spot and try_build(building_type, spot):
                    continue

        # Extra complete Bases are the strongest survival insurance. Build them
        # once we can afford to keep producing, and earlier when the only Base is
        # damaged or enemies are near it.
        desired_bases = 1 + (world.turn >= 35) + (world.turn >= 130)
        if threatened:
            desired_bases += 1
        desired_bases = min(desired_bases, 3)
        while len(bases) + self._count_planned(actions, "Base") < desired_bases:
            spot = self._best_base_tile(world, reserved)
            if spot is None or not try_build("Base", spot):
                break

        desired_barracks = 1 + (world.turn >= 70)
        desired_factories = (world.turn >= 35) + (world.turn >= 140)
        desired_airbases = (world.turn >= 115)
        desired_mines = min(10, 2 + world.turn // 25 + len(complete_bases))
        if own_unit_count < 4 and world.turn < 40:
            desired_mines = min(desired_mines, 4)

        build_sequence: list[str] = []
        build_sequence.extend(["Barracks"] * max(0, desired_barracks - len(barracks)))
        build_sequence.extend(["Factory"] * max(0, desired_factories - len(factories)))
        build_sequence.extend(["Airbase"] * max(0, desired_airbases - len(airbases)))
        build_sequence.extend(["Mine"] * max(0, desired_mines - len(mines)))

        # If enemies are visible and production is thin, prioritize production
        # before squeezing out another mine.
        if world.enemies and len(barracks) + len(factories) < 3:
            build_sequence = sorted(
                build_sequence, key=lambda t: 0 if t in ("Barracks", "Factory") else 1
            )

        for building_type in build_sequence:
            spot = self._best_adjacent_build_tile(
                world, complete, reserved, building_type
            )
            if spot is None:
                continue
            try_build(building_type, spot)

        return actions, gold, reserved

    def _best_adjacent_build_tile(
        self,
        world: World,
        anchors: list[dict],
        reserved: set[HexCoord],
        building_type: str,
    ) -> HexCoord | None:
        candidates: list[tuple[tuple, HexCoord]] = []
        bases = [b for b in world.own_buildings if b["type"] == "Base"]
        base_coords = [self._coord(b) for b in bases]
        enemy_coords = [self._coord(e) for e in world.enemies]

        for anchor in anchors:
            ac = self._coord(anchor)
            for coord in world.grid.neighbors(ac):
                if coord in reserved or coord not in world.visible:
                    continue
                if building_type != "Base" and not any(
                    world.grid.distance(coord, self._coord(b)) <= 1
                    for b in anchors
                ):
                    continue
                terrain = self._terrain(world, coord)
                rich_bonus = 1 if terrain == "rich_resource" else 0
                enemy_dist = min(
                    (world.grid.distance(coord, ec) for ec in enemy_coords), default=99
                )
                base_dist = min(
                    (world.grid.distance(coord, bc) for bc in base_coords), default=0
                )
                if building_type in _RESOURCE_BUILDINGS:
                    score = (-rich_bonus, -enemy_dist, base_dist, coord.r, coord.q)
                elif building_type in _PRODUCTION_BUILDINGS:
                    spawn_room = self._free_neighbor_count(world, coord, reserved)
                    score = (-spawn_room, -enemy_dist, base_dist, coord.r, coord.q)
                else:
                    score = (-enemy_dist, base_dist, coord.r, coord.q)
                candidates.append((score, coord))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    def _best_base_tile(self, world: World, reserved: set[HexCoord]) -> HexCoord | None:
        own_bases = [self._coord(b) for b in world.own_buildings if b["type"] == "Base"]
        enemies = [self._coord(e) for e in world.enemies]
        candidates: list[tuple[tuple, HexCoord]] = []
        for coord in world.visible:
            if coord in reserved:
                continue
            terrain = self._terrain(world, coord)
            rich = 1 if terrain == "rich_resource" else 0
            enemy_dist = min(
                (world.grid.distance(coord, ec) for ec in enemies), default=99
            )
            base_dist = min(
                (world.grid.distance(coord, bc) for bc in own_bases), default=0
            )
            spawn_room = self._free_neighbor_count(world, coord, reserved)
            if enemy_dist < 5:
                continue
            score = (-rich, -base_dist, -spawn_room, -enemy_dist, coord.r, coord.q)
            candidates.append((score, coord))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    # ── production ───────────────────────────────────────────────────────────

    def _production_actions(
        self, world: World, gold: int, reserved: set[HexCoord]
    ) -> tuple[list, int, set[HexCoord]]:
        actions: list = []
        counts = defaultdict(int)
        for unit in world.own_units:
            counts[unit["type"]] += 1
        unit_cap = self._unit_cap(world)
        if sum(counts.values()) >= unit_cap:
            return actions, gold, reserved

        producers = [
            b
            for b in world.own_buildings
            if b.get("is_complete", True) and b.get("type") in _PRODUCTION_BUILDINGS
        ]
        producers.sort(key=lambda b: (b["type"], b["r"], b["q"], b["id"]))

        for building in producers:
            if len(actions) >= _PRODUCE_LIMIT_PER_TURN:
                break
            remaining_capacity = unit_cap - sum(counts.values())
            if remaining_capacity <= 0:
                break
            slots = self._spawn_slots(world, building, reserved)
            if not slots:
                continue
            slots = slots[:remaining_capacity]
            wants = self._production_wants(world, building, counts, len(slots))
            for unit_type in wants:
                if len(actions) >= _PRODUCE_LIMIT_PER_TURN or not slots:
                    break
                cost = UNIT_STATS[unit_type].gold_cost
                if gold < cost:
                    continue
                target = slots.pop(0)
                actions.append(
                    ProduceUnitAction(
                        building_id=building["id"], unit_type=unit_type, target=target
                    )
                )
                gold -= cost
                reserved.add(target)
                counts[unit_type] += 1

        return actions, gold, reserved

    def _production_wants(
        self, world: World, building: dict, counts: dict[str, int], slots: int
    ) -> list[str]:
        btype = building["type"]
        total_units = sum(counts.values())
        threatened = self._base_threatened(world)
        wants: list[str] = []

        if btype == "Barracks":
            if counts["Scout"] < (2 if world.turn < 80 else 4):
                wants.append("Scout")
            if counts["Medic"] < max(1, counts["Infantry"] // 5) and total_units >= 4:
                wants.append("Medic")
            wants.extend(["Infantry"] * max(1, slots - len(wants)))
            if threatened:
                wants.extend(["Infantry"] * 3)
        elif btype == "Factory":
            if threatened or counts["Tank"] <= counts["Artillery"]:
                wants.append("Tank")
            wants.append("Artillery")
            wants.extend(["Tank", "Artillery"] * max(0, slots))
        elif btype == "Airbase":
            if counts["Fighter"] < counts["Bomber"] + 2:
                wants.append("Fighter")
            if world.turn > 110 or any(e["type"] == "Base" for e in world.enemies):
                wants.append("Bomber")
            wants.extend(["Fighter"] * max(0, slots - len(wants)))

        return wants[: max(1, min(slots, 3))]

    def _spawn_slots(
        self, world: World, building: dict, reserved: set[HexCoord]
    ) -> list[HexCoord]:
        coord = self._coord(building)
        candidates = [c for c in world.grid.neighbors(coord) if c not in reserved]
        candidates.sort(
            key=lambda c: (
                self._terrain(world, c) != "elevated",
                self._terrain(world, c) == "difficult",
                c.r,
                c.q,
            )
        )
        return candidates

    # ── combat and movement ──────────────────────────────────────────────────

    def _combat_and_moves(
        self, world: World, reserved: set[HexCoord]
    ) -> tuple[list, set[HexCoord]]:
        actions: list = []
        move_reserved = set(reserved)
        own_coords = {self._coord(e) for e in world.own_units + world.own_buildings}
        incoming_damage: dict[str, int] = defaultdict(int)

        for unit in sorted(
            world.own_units, key=lambda u: (u["type"], u["r"], u["q"], u["id"])
        ):
            here = self._coord(unit)
            target = self._best_attack_target(world, unit, incoming_damage, own_coords)
            if target is not None:
                tc = self._coord(target)
                actions.append(AttackAction(unit_id=unit["id"], target=tc))
                incoming_damage[target["id"]] += int(unit.get("attack_power", 0))

            dest = self._best_move(world, unit, move_reserved)
            if dest is not None and dest != here:
                actions.append(MoveAction(unit_id=unit["id"], path=[here, dest]))
                move_reserved.discard(here)
                move_reserved.add(dest)

        return actions, move_reserved

    def _best_attack_target(
        self,
        world: World,
        unit: dict,
        incoming_damage: dict[str, int],
        own_coords: set[HexCoord],
    ) -> dict | None:
        attack_range = int(unit.get("attack_range", 0))
        if attack_range <= 0:
            return None
        here = self._coord(unit)
        candidates: list[tuple[tuple, dict]] = []
        for enemy in world.enemies:
            target = self._coord(enemy)
            dist = world.grid.distance(here, target)
            if not (0 < dist <= attack_range):
                continue
            if unit["type"] == "Artillery" and self._artillery_splashes_friend(
                world, target, own_coords
            ):
                continue
            hp_left = int(enemy.get("hp", 0)) - incoming_damage[enemy["id"]]
            score = self._target_score(world, unit, enemy, dist, hp_left)
            candidates.append((score, enemy))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    def _target_score(
        self, world: World, unit: dict, target: dict, dist: int, hp_left: int
    ) -> tuple:
        ttype = target["type"]
        own_bases = [self._coord(b) for b in world.own_buildings if b["type"] == "Base"]
        target_coord = self._coord(target)
        base_dist = min(
            (world.grid.distance(target_coord, b) for b in own_bases), default=99
        )
        near_base = 0 if base_dist <= 5 else 1
        type_priority = {
            "Bomber": 0,
            "Artillery": 1,
            "Tank": 2,
            "Fighter": 3,
            "Infantry": 4,
            "Scout": 5,
            "Medic": 6,
            "Base": 7,
            "Factory": 8,
            "Airbase": 9,
            "Barracks": 10,
            "Mine": 11,
        }.get(ttype, 12)
        if unit["type"] == "Bomber" and ttype in BUILDING_STATS:
            type_priority -= 3
        killable = 0 if hp_left <= int(unit.get("attack_power", 0)) else 1
        return (
            near_base,
            type_priority,
            killable,
            max(hp_left, 0),
            dist,
            target_coord.r,
            target_coord.q,
        )

    def _best_move(
        self, world: World, unit: dict, reserved: set[HexCoord]
    ) -> HexCoord | None:
        here = self._coord(unit)
        movement = int(unit.get("movement_range", 0))
        if movement <= 0:
            return None
        reachable = self._reachable_this_turn(world, here, movement, reserved)
        if not reachable:
            return None

        if unit["type"] == "Scout":
            return self._best_scout_move(world, here, reachable)

        enemies = world.enemies
        bases = [self._coord(b) for b in world.own_buildings if b["type"] == "Base"]
        if enemies:
            attack_range = int(unit.get("attack_range", 0))
            targets = [self._coord(e) for e in enemies]
            def score(coord: HexCoord) -> tuple:
                nearest_enemy = min(world.grid.distance(coord, t) for t in targets)
                nearest_base = min(
                    (world.grid.distance(coord, b) for b in bases), default=0
                )
                in_range = 0 if attack_range and nearest_enemy <= attack_range else 1
                defend = 0 if nearest_base <= 4 else 1
                terrain_score = 0 if self._terrain(world, coord) == "elevated" else 1
                return (defend, in_range, nearest_enemy, terrain_score, coord.r, coord.q)
            return min(reachable, key=score)

        if bases:
            # Spread defenders around the nearest Base, preferring elevated tiles.
            def guard_score(coord: HexCoord) -> tuple:
                nearest_base = min(world.grid.distance(coord, b) for b in bases)
                terrain_score = 0 if self._terrain(world, coord) == "elevated" else 1
                ideal_ring = abs(nearest_base - 2)
                return (ideal_ring, terrain_score, nearest_base, coord.r, coord.q)
            best = min(reachable, key=guard_score)
            if guard_score(best) < guard_score(here):
                return best
        return None

    def _best_scout_move(
        self, world: World, here: HexCoord, reachable: set[HexCoord]
    ) -> HexCoord | None:
        own_bases = [self._coord(b) for b in world.own_buildings if b["type"] == "Base"]
        enemy_coords = [self._coord(e) for e in world.enemies]

        def score(coord: HexCoord) -> tuple:
            newly_seen = sum(
                1 for c in world.grid.disk(coord, 5) if c not in self._seen_terrain
            )
            rich_seen = sum(
                1
                for c in world.grid.disk(coord, 5)
                if self._terrain(world, c) == "rich_resource"
            )
            enemy_dist = min(
                (world.grid.distance(coord, e) for e in enemy_coords), default=99
            )
            base_dist = min(
                (world.grid.distance(coord, b) for b in own_bases), default=0
            )
            danger = 0 if enemy_dist > 2 else 1
            return (-newly_seen, -rich_seen, danger, -base_dist, coord.r, coord.q)

        best = min(reachable, key=score)
        return best if score(best) < score(here) else None

    def _reachable_this_turn(
        self,
        world: World,
        origin: HexCoord,
        movement: int,
        reserved: set[HexCoord],
    ) -> set[HexCoord]:
        blocked = {c for c in reserved if c != origin}
        costs = {
            c: 2
            for c, terrain in {**self._seen_terrain, **world.terrain}.items()
            if terrain == "difficult"
        }
        return world.grid.reachable(origin, movement, costs, blocked)

    # ── utility helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _coord(entity: dict) -> HexCoord:
        return HexCoord(int(entity["q"]), int(entity["r"]))

    @staticmethod
    def _count_planned(actions: list, building_type: str) -> int:
        return sum(
            1
            for action in actions
            if isinstance(action, ConstructBuildingAction)
            and action.building_type == building_type
        )

    def _terrain(self, world: World, coord: HexCoord) -> str:
        coord = world.grid.wrap(coord)
        return world.terrain.get(coord) or self._seen_terrain.get(coord, "normal")

    def _free_neighbor_count(
        self, world: World, coord: HexCoord, reserved: set[HexCoord]
    ) -> int:
        return sum(1 for nb in world.grid.neighbors(coord) if nb not in reserved)

    def _base_threatened(self, world: World) -> bool:
        bases = [b for b in world.own_buildings if b["type"] == "Base"]
        if any(int(b.get("hp", 999)) < int(b.get("max_hp", 999)) for b in bases):
            return True
        base_coords = [self._coord(b) for b in bases]
        for enemy in world.enemies:
            ec = self._coord(enemy)
            if any(world.grid.distance(ec, bc) <= 6 for bc in base_coords):
                return True
        return False

    def _unit_cap(self, world: World) -> int:
        cap = min(55, 10 + world.turn // 4)
        if self._base_threatened(world):
            cap += 10
        return cap

    def _artillery_splashes_friend(
        self, world: World, target: HexCoord, own_coords: set[HexCoord]
    ) -> bool:
        return any(c in own_coords for c in world.grid.ring(target, 1))
