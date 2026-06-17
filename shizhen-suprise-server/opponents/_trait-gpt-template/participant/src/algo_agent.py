"""Hybrid deterministic tactics manager for the surprise hex-grid game.

The agent is intentionally local and bounded: it uses GOAM-style macro
directives when supplied by ``llm_agent.py``, then executes all geometry,
production, movement, and combat with deterministic Python.  The search layer is
a practical HPS/MAPLE implementation for the 10 second turn budget: partial
players generate phase-level candidates, and a single shared evaluator scores
those candidates across sampled hidden-scout information states.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from math import exp

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
from engine.constants import (
    ARTILLERY_SPLASH_DAMAGE_RATIO,
    BUILDING_STATS,
    DIFFICULT_TERRAIN_MOVE_COST,
    ELEVATION_ATTACK_BONUS,
    TREATY_CUTOFF_TURN,
    UNIT_STATS,
)
from engine.entities import (
    Airbase,
    Artillery,
    Barracks,
    Base,
    Bomber,
    Factory,
    Fighter,
    Infantry,
    Medic,
    Mine,
    Scout,
    Tank,
)
from engine.hex_grid import HexCoord, HexGrid


_PRODUCTION_BUILDINGS = ("Barracks", "Factory", "Airbase")
_RESOURCE_BUILDINGS = ("Base", "Mine")
_GROUND_TYPES = ("Infantry", "Tank", "Artillery", "Scout", "Medic")
_AIR_TYPES = ("Fighter", "Bomber")
_BUILD_LIMIT_PER_TURN = 7
_PRODUCE_LIMIT_PER_TURN = 16
_MAPLE_SAMPLE_LIMIT = 4
_MAX_HPS_ROOT_CANDIDATES = 5
_DIRECTIVES = {
    "BALANCED",
    "ECONOMIC_EXPANSION",
    "SCOUT_EXPANSION",
    "DEFENSIVE_TURTLE",
    "AIR_SUPERIORITY_RUSH",
    "BASE_ASSAULT",
    "BASE_ASSAULT_AIR",
}
_DIRECTIVE_ALIASES = {
    "AIR_RUSH": "AIR_SUPERIORITY_RUSH",
    "AIR_SUPERIORITY": "AIR_SUPERIORITY_RUSH",
    "DEFENSE": "DEFENSIVE_TURTLE",
    "TURTLE": "DEFENSIVE_TURTLE",
    "ECONOMY": "ECONOMIC_EXPANSION",
    "EXPAND": "ECONOMIC_EXPANSION",
    "ASSAULT": "BASE_ASSAULT",
    "ATTACK": "BASE_ASSAULT",
    "BASE_ASSAULT_AIR_RUSH": "BASE_ASSAULT_AIR",
}


def _clamp(value: object, default: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(lo, min(hi, number))


def _default_weights(macro_directive: str) -> dict[str, float]:
    if macro_directive == "AIR_SUPERIORITY_RUSH":
        return {"air": 0.75, "ground": 0.15, "economy": 0.20, "defense": 0.45}
    if macro_directive == "BASE_ASSAULT_AIR":
        return {"air": 0.65, "ground": 0.25, "economy": 0.10, "defense": 0.25}
    if macro_directive == "BASE_ASSAULT":
        return {"air": 0.20, "ground": 0.65, "economy": 0.10, "defense": 0.25}
    if macro_directive == "DEFENSIVE_TURTLE":
        return {"air": 0.30, "ground": 0.45, "economy": 0.20, "defense": 0.85}
    if macro_directive == "SCOUT_EXPANSION":
        return {"air": 0.10, "ground": 0.35, "economy": 0.45, "defense": 0.30}
    if macro_directive == "ECONOMIC_EXPANSION":
        return {"air": 0.10, "ground": 0.25, "economy": 0.75, "defense": 0.25}
    return {"air": 0.25, "ground": 0.40, "economy": 0.35, "defense": 0.40}


@dataclass(frozen=True)
class StrategyDirective:
    macro_directive: str = "BALANCED"
    resource_weights: dict[str, float] = field(
        default_factory=lambda: _default_weights("BALANCED")
    )
    priority_targets: tuple[str, ...] = ()
    aggression: float = 0.45
    risk_tolerance: float = 0.35

    @classmethod
    def from_payload(cls, payload: object | None) -> "StrategyDirective":
        if isinstance(payload, StrategyDirective):
            return payload
        if not isinstance(payload, dict):
            return cls()

        raw_macro = str(
            payload.get("macro_directive")
            or payload.get("directive")
            or payload.get("strategy")
            or "BALANCED"
        ).upper()
        macro = _DIRECTIVE_ALIASES.get(raw_macro, raw_macro)
        if macro not in _DIRECTIVES:
            macro = "BALANCED"

        weights = _default_weights(macro)
        raw_weights = payload.get("resource_weights") or payload.get(
            "resource_allocation"
        )
        if isinstance(raw_weights, dict):
            for key in ("air", "ground", "economy", "defense"):
                if key in raw_weights:
                    weights[key] = _clamp(raw_weights[key], weights[key])

        raw_targets = payload.get("priority_targets") or ()
        if isinstance(raw_targets, str):
            targets = (raw_targets,)
        elif isinstance(raw_targets, list | tuple):
            targets = tuple(str(t) for t in raw_targets if isinstance(t, str))
        else:
            targets = ()

        default_aggression = {
            "ECONOMIC_EXPANSION": 0.25,
            "SCOUT_EXPANSION": 0.35,
            "DEFENSIVE_TURTLE": 0.20,
            "AIR_SUPERIORITY_RUSH": 0.55,
            "BASE_ASSAULT": 0.80,
            "BASE_ASSAULT_AIR": 0.85,
        }.get(macro, 0.45)

        return cls(
            macro_directive=macro,
            resource_weights=weights,
            priority_targets=targets,
            aggression=_clamp(payload.get("aggression"), default_aggression),
            risk_tolerance=_clamp(payload.get("risk_tolerance"), 0.35),
        )


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


@dataclass
class InfluenceMaps:
    grid: HexGrid
    threat: list[list[float]]
    protection: list[list[float]]
    concealment: list[list[float]]

    def value(self, layer: str, coord: HexCoord) -> float:
        matrix = getattr(self, layer)
        wrapped = self.grid.wrap(coord)
        col = (wrapped.q + wrapped.r // 2) % self.grid.width
        return matrix[wrapped.r % self.grid.height][col]


@dataclass(frozen=True)
class WorldSample:
    hidden_scouts: tuple[HexCoord, ...] = ()


@dataclass(frozen=True)
class PendingProduction:
    building_id: str
    unit_type: str
    target: HexCoord
    due_turn: int


@dataclass
class PortfolioCandidate:
    production_player: str
    movement_player: str
    combat_player: str
    actions: list
    reserved: set[HexCoord]
    gold: int
    value: float = 0.0

    @property
    def name(self) -> str:
        return f"{self.production_player}/{self.movement_player}/{self.combat_player}"


class AlgoAgent(PlayerAgent):
    def __init__(self) -> None:
        self._seen_terrain: dict[HexCoord, str] = {}
        self._last_seen_enemy: dict[str, tuple[HexCoord, int, str]] = {}
        self._proposed_peace: set[str] = set()
        self._pending_production: list[PendingProduction] = []
        self._distance_cache: dict[tuple[int, int, int, int, int, int], int] = {}
        self._disk_cache: dict[tuple[int, int, int, int, int], tuple[HexCoord, ...]] = {}
        self._turn_cache: dict[str, object] = {}
        self._movement_cost_cache: dict[HexCoord, int] | None = None
        self._trait = os.environ.get("BOT_TRAIT", "adaptive").strip().lower()

    async def decide(
        self, observation: dict, directive: StrategyDirective | dict | None = None
    ) -> ActionPayload:
        world = self._parse_world(observation)
        self._turn_cache = {}
        self._movement_cost_cache = None
        self._reconcile_pending_production(world)
        strategy = (
            StrategyDirective.from_payload(directive)
            if directive is not None
            else self._local_directive(world)
        )

        actions: list = []
        actions.extend(self._diplomacy_actions(world))
        if not world.enemies:
            candidate = self._quiet_turn_candidate(world, strategy)
            actions.extend(candidate.actions)
            self._remember_production_orders(world, candidate.actions)
            return ActionPayload(
                player_id=world.pid, turn_number=world.turn, actions=actions
            )

        influence = self._build_influence_maps(world, strategy)
        candidate = self._select_hps_maple_candidate(world, strategy, influence)
        actions.extend(candidate.actions)
        self._remember_production_orders(world, candidate.actions)

        return ActionPayload(
            player_id=world.pid, turn_number=world.turn, actions=actions
        )

    # -- world parsing -----------------------------------------------------

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

    def _local_directive(self, world: World) -> StrategyDirective:
        counts = self._unit_counts(world)
        enemy_types = {e.get("type") for e in world.enemies}
        visible_enemy_base = any(e.get("type") == "Base" for e in world.enemies)
        finishable_base = self._finishable_enemy_base(world)
        base_threat = self._base_threatened(world)
        mines = sum(1 for b in world.own_buildings if b.get("type") == "Mine")

        if finishable_base is not None:
            return StrategyDirective.from_payload(
                {
                    "macro_directive": (
                        "BASE_ASSAULT_AIR" if counts["Bomber"] else "BASE_ASSAULT"
                    ),
                    "priority_targets": [finishable_base["id"], "Base"],
                    "resource_weights": {"air": 0.55, "ground": 0.55, "economy": 0.05},
                    "aggression": 0.95,
                    "risk_tolerance": 0.65,
                }
            )
        if "Bomber" in enemy_types or (
            "Airbase" in enemy_types and counts["Fighter"] < 2
        ) or (
            "Airbase" in self._recent_enemy_types(world, max_age=18)
            and counts["Fighter"] < 3
        ):
            return StrategyDirective.from_payload(
                {
                    "macro_directive": "AIR_SUPERIORITY_RUSH",
                    "priority_targets": ["Bomber", "Airbase"],
                    "resource_weights": {"air": 0.75, "ground": 0.2, "defense": 0.6},
                    "aggression": 0.55,
                    "risk_tolerance": 0.30,
                }
            )
        if base_threat:
            return StrategyDirective.from_payload(
                {
                    "macro_directive": "DEFENSIVE_TURTLE",
                    "priority_targets": ["Bomber", "Artillery", "Tank"],
                    "resource_weights": {"defense": 0.9, "ground": 0.55},
                    "aggression": 0.25,
                    "risk_tolerance": 0.20,
                }
            )
        trait_strategy = self._trait_directive(world, counts, mines, visible_enemy_base)
        if trait_strategy is not None:
            return trait_strategy
        if "Factory" in self._recent_enemy_types(world, max_age=14) and counts["Tank"] < 4:
            return StrategyDirective.from_payload(
                {
                    "macro_directive": "DEFENSIVE_TURTLE",
                    "priority_targets": ["Artillery", "Factory", "Tank"],
                    "resource_weights": {"defense": 0.75, "ground": 0.65},
                    "aggression": 0.35,
                    "risk_tolerance": 0.25,
                }
            )
        if visible_enemy_base and (counts["Bomber"] or world.turn >= 130):
            return StrategyDirective.from_payload(
                {
                    "macro_directive": "BASE_ASSAULT_AIR",
                    "priority_targets": ["Base", "Airbase", "Factory"],
                    "aggression": 0.85,
                    "risk_tolerance": 0.55,
                }
            )
        if world.turn < 55 and mines < 5:
            return StrategyDirective.from_payload(
                {"macro_directive": "ECONOMIC_EXPANSION"}
            )
        if counts["Scout"] < (2 if world.turn < 120 else 4):
            return StrategyDirective.from_payload({"macro_directive": "SCOUT_EXPANSION"})
        if visible_enemy_base:
            return StrategyDirective.from_payload(
                {"macro_directive": "BASE_ASSAULT", "priority_targets": ["Base"]}
            )
        return StrategyDirective()

    def _trait_directive(
        self,
        world: World,
        counts: dict[str, int],
        mines: int,
        visible_enemy_base: bool,
    ) -> StrategyDirective | None:
        trait = self._trait
        if trait in ("", "adaptive"):
            return None
        if trait in ("balanced", "generalist"):
            return StrategyDirective()
        if trait in ("resource", "rich-resource", "economy"):
            return StrategyDirective.from_payload(
                {
                    "macro_directive": "ECONOMIC_EXPANSION",
                    "resource_weights": {
                        "economy": 0.9,
                        "defense": 0.35,
                        "ground": 0.25,
                        "air": 0.1,
                    },
                    "aggression": 0.2,
                    "risk_tolerance": 0.25,
                }
            )
        if trait == "diplomacy":
            return StrategyDirective.from_payload(
                {
                    "macro_directive": "ECONOMIC_EXPANSION",
                    "resource_weights": {
                        "economy": 0.75,
                        "defense": 0.65,
                        "ground": 0.25,
                        "air": 0.15,
                    },
                    "aggression": 0.15,
                    "risk_tolerance": 0.18,
                }
            )
        if trait in ("defense", "fortress", "expansion-fortress"):
            return StrategyDirective.from_payload(
                {
                    "macro_directive": "DEFENSIVE_TURTLE",
                    "priority_targets": ["Bomber", "Artillery", "Tank"],
                    "resource_weights": {
                        "defense": 0.95,
                        "ground": 0.55,
                        "economy": 0.35,
                    },
                    "aggression": 0.22,
                    "risk_tolerance": 0.18,
                }
            )
        if trait in ("ground-rush", "rush"):
            return StrategyDirective.from_payload(
                {
                    "macro_directive": "BASE_ASSAULT",
                    "priority_targets": ["Base", "Factory", "Airbase"],
                    "resource_weights": {"ground": 0.85, "economy": 0.1},
                    "aggression": 0.82,
                    "risk_tolerance": 0.55,
                }
            )
        if trait in ("air-control", "air"):
            return StrategyDirective.from_payload(
                {
                    "macro_directive": "AIR_SUPERIORITY_RUSH",
                    "priority_targets": ["Bomber", "Airbase", "Fighter"],
                    "resource_weights": {"air": 0.85, "defense": 0.45},
                    "aggression": 0.58,
                    "risk_tolerance": 0.35,
                }
            )
        if trait in ("scout-vision", "scout"):
            return StrategyDirective.from_payload(
                {
                    "macro_directive": "SCOUT_EXPANSION",
                    "priority_targets": ["Scout", "Mine"],
                    "resource_weights": {"economy": 0.55, "ground": 0.35},
                    "aggression": 0.32,
                    "risk_tolerance": 0.32,
                }
            )
        if trait in ("base-assault", "assault"):
            return StrategyDirective.from_payload(
                {
                    "macro_directive": "BASE_ASSAULT_AIR"
                    if counts["Bomber"] or world.turn >= 120 or visible_enemy_base
                    else "BASE_ASSAULT",
                    "priority_targets": ["Base", "Airbase", "Factory"],
                    "resource_weights": {"air": 0.55, "ground": 0.65},
                    "aggression": 0.9,
                    "risk_tolerance": 0.62,
                }
            )
        return None

    # -- influence maps ----------------------------------------------------

    def _build_influence_maps(
        self, world: World, strategy: StrategyDirective
    ) -> InfluenceMaps:
        threat = self._zero_matrix(world.grid)
        protection = self._zero_matrix(world.grid)
        concealment = self._zero_matrix(world.grid)

        if world.enemies or len(world.own_units) < 8:
            for coord in world.grid.all_coords():
                terrain = self._terrain(world, coord)
                if terrain == "concealment":
                    self._matrix_add(world.grid, concealment, coord, 1.0)

        for enemy in world.enemies:
            coord = self._coord(enemy)
            etype = enemy.get("type", "")
            if etype in UNIT_STATS:
                stats = UNIT_STATS[etype]
                radius = max(stats.attack_range + stats.movement_range, 1)
                if etype == "Artillery":
                    radius += 2
                if etype == "Bomber":
                    radius += 2
                base_power = float(stats.attack_power)
                if etype == "Bomber":
                    base_power *= Bomber.BUILDING_DAMAGE_MULTIPLIER
                for tile in world.grid.disk(coord, min(radius + 2, 8)):
                    dist = world.grid.distance(coord, tile)
                    if etype == "Artillery":
                        value = base_power * exp(-0.55 * max(0, dist - 1))
                    else:
                        value = base_power * exp(-0.75 * max(0, dist - stats.attack_range))
                    self._matrix_add(world.grid, threat, tile, value)
            elif etype in BUILDING_STATS:
                radius = 5 if etype in ("Base", "Airbase", "Factory") else 2
                value = {"Base": 12.0, "Airbase": 22.0, "Factory": 18.0}.get(
                    etype, 4.0
                )
                for tile in world.grid.disk(coord, radius):
                    dist = world.grid.distance(coord, tile)
                    self._matrix_add(world.grid, threat, tile, value * exp(-0.45 * dist))

        visible_enemy_ids = {enemy["id"] for enemy in world.enemies}
        for eid, (coord, seen_turn, etype) in self._last_seen_enemy.items():
            age = world.turn - seen_turn
            if eid in visible_enemy_ids or age < 1 or age > 24:
                continue
            decay = exp(-0.16 * age)
            if etype in UNIT_STATS:
                stats = UNIT_STATS[etype]
                radius = min(8, stats.attack_range + stats.movement_range + min(age, 4))
                base_power = float(stats.attack_power) * decay
                if etype == "Bomber":
                    base_power *= Bomber.BUILDING_DAMAGE_MULTIPLIER
                for tile in world.grid.disk(coord, max(1, radius)):
                    dist = world.grid.distance(coord, tile)
                    self._matrix_add(
                        world.grid,
                        threat,
                        tile,
                        base_power * exp(-0.65 * max(0, dist - stats.attack_range)),
                    )
            elif etype in ("Airbase", "Factory", "Base"):
                value = {"Base": 10.0, "Airbase": 28.0, "Factory": 22.0}[etype] * decay
                for tile in world.grid.disk(coord, 6):
                    dist = world.grid.distance(coord, tile)
                    self._matrix_add(world.grid, threat, tile, value * exp(-0.4 * dist))

        defense_weight = strategy.resource_weights.get("defense", 0.4)
        for building in world.own_buildings:
            if building.get("type") != "Base":
                continue
            coord = self._coord(building)
            hp_ratio = self._hp_ratio(building)
            for tile in world.grid.disk(coord, 6):
                dist = world.grid.distance(coord, tile)
                value = (7 - dist) * (8.0 + 12.0 * defense_weight)
                value *= 1.0 + (1.0 - hp_ratio)
                self._matrix_add(world.grid, protection, tile, value)

        for unit in world.own_units:
            coord = self._coord(unit)
            if unit.get("type") == "Medic":
                for tile in world.grid.disk(coord, 2):
                    dist = world.grid.distance(coord, tile)
                    self._matrix_add(
                        world.grid,
                        protection,
                        tile,
                        max(0.0, Medic.HEAL_AMOUNT * (1.5 - 0.45 * dist)),
                    )
            elif unit.get("type") in ("Fighter", "Tank"):
                for tile in world.grid.disk(coord, 2):
                    self._matrix_add(world.grid, protection, tile, 5.0)

        return InfluenceMaps(
            grid=world.grid,
            threat=threat,
            protection=protection,
            concealment=concealment,
        )

    @staticmethod
    def _zero_matrix(grid: HexGrid) -> list[list[float]]:
        return [[0.0 for _ in range(grid.width)] for _ in range(grid.height)]

    @staticmethod
    def _matrix_add(
        grid: HexGrid, matrix: list[list[float]], coord: HexCoord, value: float
    ) -> None:
        wrapped = grid.wrap(coord)
        col = (wrapped.q + wrapped.r // 2) % grid.width
        matrix[wrapped.r % grid.height][col] += value

    # -- HPS / MAPLE root search ------------------------------------------

    def _quiet_turn_candidate(
        self, world: World, strategy: StrategyDirective
    ) -> PortfolioCandidate:
        influence = InfluenceMaps(
            grid=world.grid,
            threat=self._zero_matrix(world.grid),
            protection=self._zero_matrix(world.grid),
            concealment=self._zero_matrix(world.grid),
        )
        production_player = self._phase_portfolios(strategy)[0][0]
        reserved = set(world.occupied) | self._pending_spawn_reservations(world)
        scout_actions, reserved = self._quiet_scout_moves(world, reserved)
        build_actions, gold, reserved = self._build_actions(
            world,
            world.gold,
            reserved,
            influence,
            strategy,
            production_player=production_player,
        )
        produce_actions, gold, reserved = self._production_actions(
            world,
            gold,
            reserved,
            strategy,
            production_player=production_player,
        )
        return PortfolioCandidate(
            production_player=production_player,
            movement_player="scout",
            combat_player="focus_fire",
            actions=scout_actions + build_actions + produce_actions,
            reserved=reserved,
            gold=gold,
        )

    def _quiet_scout_moves(
        self, world: World, reserved: set[HexCoord]
    ) -> tuple[list, set[HexCoord]]:
        actions: list = []
        scouts = sorted(
            (u for u in world.own_units if u.get("type") == "Scout"),
            key=lambda u: (u["r"], u["q"], u["id"]),
        )[:3]
        bases = [self._coord(b) for b in world.own_buildings if b.get("type") == "Base"]

        for scout in scouts:
            here = self._coord(scout)
            reachable = self._reachable_this_turn(
                world, here, int(scout.get("movement_range", 0)), reserved
            )
            if not reachable:
                continue

            def score(coord: HexCoord) -> tuple:
                scan = self._disk(world, coord, 2)
                newly_seen = sum(1 for c in scan if c not in self._seen_terrain)
                rich_seen = sum(
                    1 for c in scan if self._terrain(world, c) == "rich_resource"
                )
                base_dist = min(
                    (self._distance(world, coord, b) for b in bases), default=0
                )
                return (-newly_seen, -rich_seen, -base_dist, coord.r, coord.q)

            dest = min(reachable, key=score)
            if dest == here or score(dest) >= score(here):
                continue
            path = self._movement_path(
                world, here, dest, int(scout.get("movement_range", 0)), reserved
            )
            if path and len(path) > 1:
                actions.append(MoveAction(unit_id=scout["id"], path=path))
                reserved.discard(here)
                reserved.add(path[-1])
        return actions, reserved

    def _select_hps_maple_candidate(
        self, world: World, strategy: StrategyDirective, influence: InfluenceMaps
    ) -> PortfolioCandidate:
        production_players, movement_players, combat_players = self._phase_portfolios(
            strategy
        )
        if not world.enemies:
            combat_players = combat_players[:1]
            movement_players = movement_players[:1]
        if len(world.own_units) >= 24:
            production_players = production_players[:1]
            movement_players = movement_players[:1]
            combat_players = combat_players[:1]
        if len(world.own_units) >= 12:
            production_players = production_players[:2]
            movement_players = movement_players[:1]
            combat_players = combat_players[:2]
        root_combinations = list(
            product(production_players, movement_players, combat_players)
        )[:_MAX_HPS_ROOT_CANDIDATES]
        if len(world.own_units) >= 8 or not world.enemies:
            root_combinations = root_combinations[:1]
        elif len(root_combinations) > 2:
            root_combinations = root_combinations[:2]

        samples = (
            self._sample_hidden_scout_states(world, influence)
            if len(root_combinations) > 1
            else [WorldSample()]
        )

        candidates: list[PortfolioCandidate] = []
        base_reserved = set(world.occupied) | self._pending_spawn_reservations(world)
        for production_player, movement_player, combat_player in root_combinations:
            reserved = set(base_reserved)
            unit_actions, reserved = self._combat_and_moves(
                world,
                reserved,
                influence,
                strategy,
                movement_player=movement_player,
                combat_player=combat_player,
            )
            build_actions, gold, reserved = self._build_actions(
                world,
                world.gold,
                reserved,
                influence,
                strategy,
                production_player=production_player,
            )
            produce_actions, gold, reserved = self._production_actions(
                world,
                gold,
                reserved,
                strategy,
                production_player=production_player,
            )
            candidate = PortfolioCandidate(
                production_player=production_player,
                movement_player=movement_player,
                combat_player=combat_player,
                actions=unit_actions + build_actions + produce_actions,
                reserved=reserved,
                gold=gold,
            )
            candidate.value = (
                self._maple_value_candidate(world, strategy, influence, candidate, samples)
                if len(root_combinations) > 1
                else 0.0
            )
            candidates.append(candidate)

        if not candidates:
            return PortfolioCandidate(
                "balanced", "guard", "focus_fire", [], set(), world.gold
            )
        return max(candidates, key=lambda c: (c.value, -len(c.actions), c.name))

    def _phase_portfolios(
        self, strategy: StrategyDirective
    ) -> tuple[list[str], list[str], list[str]]:
        macro = strategy.macro_directive
        if macro == "AIR_SUPERIORITY_RUSH":
            return (
                ["anti_air", "air_rush", "balanced"],
                ["intercept", "guard", "scout"],
                ["anti_bomber", "focus_fire", "artillery_cluster"],
            )
        if macro == "BASE_ASSAULT_AIR":
            return (
                ["air_rush", "ground_assault", "balanced"],
                ["assault", "intercept", "scout"],
                ["base_snipe", "anti_bomber", "focus_fire"],
            )
        if macro == "BASE_ASSAULT":
            return (
                ["ground_assault", "balanced", "economy"],
                ["assault", "guard", "scout"],
                ["base_snipe", "artillery_cluster", "focus_fire"],
            )
        if macro == "DEFENSIVE_TURTLE":
            return (
                ["defense", "anti_air", "balanced"],
                ["guard", "intercept", "medic_pulse"],
                ["anti_bomber", "focus_fire", "artillery_cluster"],
            )
        if macro == "SCOUT_EXPANSION":
            return (
                ["economy", "balanced", "ground_assault"],
                ["scout", "guard", "assault"],
                ["focus_fire", "artillery_cluster", "anti_bomber"],
            )
        if macro == "ECONOMIC_EXPANSION":
            return (
                ["economy", "balanced", "defense"],
                ["scout", "guard", "medic_pulse"],
                ["focus_fire", "anti_bomber", "artillery_cluster"],
            )
        return (
            ["balanced", "economy", "defense"],
            ["guard", "scout", "assault"],
            ["focus_fire", "anti_bomber", "artillery_cluster"],
        )

    def _sample_hidden_scout_states(
        self, world: World, influence: InfluenceMaps
    ) -> list[WorldSample]:
        candidates: set[HexCoord] = set()
        for coord, terrain in self._seen_terrain.items():
            wrapped = world.grid.wrap(coord)
            if terrain == "concealment" and wrapped not in world.occupied:
                candidates.add(wrapped)

        for coord, seen_turn, entity_type in self._last_seen_enemy.values():
            if entity_type == "Scout" and world.turn - seen_turn <= 30:
                wrapped = world.grid.wrap(coord)
                if wrapped not in world.occupied:
                    candidates.add(wrapped)

        own_bases = [self._coord(b) for b in world.own_buildings if b["type"] == "Base"]

        def scout_likelihood(coord: HexCoord) -> tuple:
            base_dist = min((world.grid.distance(coord, b) for b in own_bases), default=20)
            conceal = influence.value("concealment", coord)
            threat = influence.value("threat", coord)
            visible_penalty = 1 if coord in world.visible else 0
            return (visible_penalty, base_dist, -conceal, threat, coord.r, coord.q)

        ranked = sorted(candidates, key=scout_likelihood)[: _MAPLE_SAMPLE_LIMIT - 1]
        samples = [WorldSample()]
        samples.extend(WorldSample((coord,)) for coord in ranked)
        return samples

    def _maple_value_candidate(
        self,
        world: World,
        strategy: StrategyDirective,
        influence: InfluenceMaps,
        candidate: PortfolioCandidate,
        samples: list[WorldSample],
    ) -> float:
        values: list[float] = []
        for sample in samples:
            if self._candidate_bumps_hidden_scout(candidate, sample):
                continue
            values.append(
                self._evaluate_candidate(world, strategy, influence, candidate, sample)
            )
        if not values:
            return -1_000_000.0
        return sum(values) / len(values) + 0.25 * len(values)

    @staticmethod
    def _candidate_bumps_hidden_scout(
        candidate: PortfolioCandidate, sample: WorldSample
    ) -> bool:
        hidden = set(sample.hidden_scouts)
        if not hidden:
            return False
        for action in candidate.actions:
            if isinstance(action, MoveAction) and action.path:
                if action.path[-1] in hidden:
                    return True
        return False

    def _evaluate_candidate(
        self,
        world: World,
        strategy: StrategyDirective,
        influence: InfluenceMaps,
        candidate: PortfolioCandidate,
        sample: WorldSample,
    ) -> float:
        score = 0.0
        score += (world.gold - candidate.gold) * 0.035
        score -= candidate.gold * 0.004

        action_targets = {
            (action.target.q, action.target.r)
            for action in candidate.actions
            if isinstance(action, AttackAction)
        }
        entity_by_coord = {(e["q"], e["r"]): e for e in world.enemies}

        for action in candidate.actions:
            if isinstance(action, ConstructBuildingAction):
                score += self._building_value(action.building_type, strategy)
                score -= influence.value("threat", action.coord) * 0.012
                score += influence.value("protection", action.coord) * 0.006
                if self._terrain(world, action.coord) == "rich_resource":
                    score += 10.0 if action.building_type in _RESOURCE_BUILDINGS else 2.0
            elif isinstance(action, ProduceUnitAction):
                score += self._unit_value(action.unit_type, strategy)
                score -= influence.value("threat", action.target) * 0.004
            elif isinstance(action, AttackAction):
                target = entity_by_coord.get((action.target.q, action.target.r))
                if target:
                    attacker = self._own_unit_by_id(world, action.unit_id)
                    score += self._attack_value(world, attacker, target, strategy)
            elif isinstance(action, MoveAction) and action.path:
                unit = self._own_unit_by_id(world, action.unit_id)
                dest = action.path[-1]
                score += influence.value("protection", dest) * 0.012
                score += influence.value("concealment", dest) * (
                    3.5 if unit and unit.get("type") == "Scout" else 0.6
                )
                score -= influence.value("threat", dest) * (
                    0.012 * (1.0 - strategy.risk_tolerance)
                )
                if unit:
                    score += self._movement_value(world, strategy, unit, dest)

        for hidden in sample.hidden_scouts:
            own_bases = [
                self._coord(b) for b in world.own_buildings if b.get("type") == "Base"
            ]
            near_base = min(
                (world.grid.distance(hidden, base) for base in own_bases), default=99
            )
            score -= max(0, 8 - near_base) * 1.5

        if strategy.macro_directive.startswith("BASE_ASSAULT"):
            score += 1.25 * len(action_targets)
        if strategy.macro_directive == "DEFENSIVE_TURTLE":
            score += sum(
                1
                for action in candidate.actions
                if isinstance(action, MoveAction)
                and self._near_own_base(world, action.path[-1], radius=4)
            )
        return score

    # -- diplomacy ---------------------------------------------------------

    def _diplomacy_actions(self, world: World) -> list:
        if world.turn >= min(TREATY_CUTOFF_TURN, world.max_turns):
            return [
                BreakTreatyAction(partner_player_id=pid, treaty_type="peace")
                for pid in sorted(world.peace_players)
            ]

        actions: list = []
        imminent_hostiles = self._imminent_hostile_players(world)
        finishable_base = self._finishable_enemy_base(world)
        finish_owner = finishable_base.get("owner_id") if finishable_base else None

        for pid in sorted(world.peace_players):
            if pid == finish_owner or (world.turn >= 165 and pid in imminent_hostiles):
                actions.append(BreakTreatyAction(partner_player_id=pid, treaty_type="peace"))

        aggressive_trait = self._trait in {
            "ground-rush",
            "rush",
            "base-assault",
            "assault",
        }
        diplomacy_trait = self._trait == "diplomacy"
        accept_until = 195 if diplomacy_trait else 80 if aggressive_trait else 185
        proposal_limit = 18 if diplomacy_trait else 4 if aggressive_trait else 12

        for proposal in world.incoming_treaty_proposals:
            proposer = proposal.get("proposer_id")
            if proposer:
                accept = (
                    world.turn < accept_until
                    and proposer != finish_owner
                )
                actions.append(
                    RespondTreatyAction(
                        proposing_player_id=proposer,
                        treaty_type=proposal.get("treaty_type", "peace"),
                        accept=accept,
                    )
                )

        if finishable_base is not None or (
            aggressive_trait and world.turn >= accept_until
        ):
            return actions[:8]

        for pid in sorted(world.known_players):
            if (
                pid != world.pid
                and pid not in world.peace_players
                and pid not in self._proposed_peace
                and world.turn < accept_until
                and len(self._proposed_peace) < proposal_limit
            ):
                actions.append(ProposeTreatyAction(target_player_id=pid))
                self._proposed_peace.add(pid)
        return actions[:8]

    # -- production and construction --------------------------------------

    def _build_actions(
        self,
        world: World,
        gold: int,
        reserved: set[HexCoord],
        influence: InfluenceMaps,
        strategy: StrategyDirective,
        production_player: str,
    ) -> tuple[list, int, set[HexCoord]]:
        actions: list = []
        complete = [b for b in world.own_buildings if b.get("is_complete", True)]
        bases = [b for b in world.own_buildings if b["type"] == "Base"]
        complete_bases = [b for b in bases if b.get("is_complete", True)]
        mines = [b for b in world.own_buildings if b["type"] == "Mine"]
        barracks = [b for b in world.own_buildings if b["type"] == "Barracks"]
        factories = [b for b in world.own_buildings if b["type"] == "Factory"]
        airbases = [b for b in world.own_buildings if b["type"] == "Airbase"]
        counts = self._unit_counts(world)
        threatened = self._base_threatened(world)
        weights = strategy.resource_weights

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

        if world.turn <= 1 and complete:
            opening = (
                ("Airbase",)
                if production_player in ("air_rush", "anti_air")
                and gold >= BUILDING_STATS["Airbase"].gold_cost
                else ("Barracks", "Mine", "Mine")
            )
            for building_type in opening:
                spot = self._best_adjacent_build_tile(
                    world, complete, reserved, influence, building_type
                )
                if spot:
                    try_build(building_type, spot)

        desired_bases = 1 + (world.turn >= 28) + (world.turn >= 85) + (world.turn >= 145)
        if threatened or production_player == "defense":
            desired_bases += 1
        desired_bases = min(desired_bases, 4)
        while len(bases) + self._count_planned(actions, "Base") < desired_bases:
            spot = self._best_base_tile(world, reserved, influence)
            if spot is None or not try_build("Base", spot):
                break

        desired_barracks = 1 + (world.turn >= 65) + (production_player == "defense")
        desired_factories = (
            (world.turn >= 32)
            + (world.turn >= 120)
            + (production_player == "ground_assault")
        )
        desired_airbases = (
            (world.turn >= 100)
            + (weights.get("air", 0.0) >= 0.55)
            + (production_player in ("air_rush", "anti_air"))
        )
        desired_mines = min(
            12,
            2
            + world.turn // 24
            + len(complete_bases)
            + int(2 * weights.get("economy", 0.0))
            + (production_player == "economy"),
        )
        if len(world.own_units) < 4 and world.turn < 45:
            desired_mines = min(desired_mines, 4)
        if threatened and counts["Infantry"] + counts["Tank"] < 8:
            desired_mines = min(desired_mines, 5)

        build_sequence: list[str] = []
        build_sequence.extend(["Barracks"] * max(0, desired_barracks - len(barracks)))
        build_sequence.extend(["Factory"] * max(0, desired_factories - len(factories)))
        build_sequence.extend(["Airbase"] * max(0, desired_airbases - len(airbases)))
        build_sequence.extend(["Mine"] * max(0, desired_mines - len(mines)))

        priority_order = {
            "air_rush": {"Airbase": 0, "Factory": 1, "Barracks": 2, "Mine": 3},
            "anti_air": {"Airbase": 0, "Barracks": 1, "Factory": 2, "Mine": 3},
            "ground_assault": {"Factory": 0, "Barracks": 1, "Airbase": 2, "Mine": 3},
            "defense": {"Barracks": 0, "Factory": 1, "Airbase": 2, "Mine": 3},
            "economy": {"Mine": 0, "Barracks": 1, "Factory": 2, "Airbase": 3},
            "balanced": {"Barracks": 0, "Mine": 1, "Factory": 2, "Airbase": 3},
        }.get(production_player, {})
        build_sequence.sort(key=lambda t: (priority_order.get(t, 9), t))

        for building_type in build_sequence:
            spot = self._best_adjacent_build_tile(
                world, complete, reserved, influence, building_type
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
        influence: InfluenceMaps,
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
                    world.grid.distance(coord, self._coord(b)) <= 1 for b in anchors
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
                threat = influence.value("threat", coord)
                protection = influence.value("protection", coord)
                if building_type in _RESOURCE_BUILDINGS:
                    score = (-rich_bonus, threat, -enemy_dist, base_dist, coord.r, coord.q)
                elif building_type in _PRODUCTION_BUILDINGS:
                    spawn_room = self._free_neighbor_count(world, coord, reserved)
                    score = (
                        -spawn_room,
                        threat,
                        -protection,
                        -enemy_dist,
                        base_dist,
                        coord.r,
                        coord.q,
                    )
                else:
                    score = (threat, -enemy_dist, base_dist, coord.r, coord.q)
                candidates.append((score, coord))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    def _best_base_tile(
        self, world: World, reserved: set[HexCoord], influence: InfluenceMaps
    ) -> HexCoord | None:
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
            threat = influence.value("threat", coord)
            if enemy_dist < 5 or threat > 90:
                continue
            score = (threat, -rich, -base_dist, -spawn_room, -enemy_dist, coord.r, coord.q)
            candidates.append((score, coord))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    def _production_actions(
        self,
        world: World,
        gold: int,
        reserved: set[HexCoord],
        strategy: StrategyDirective,
        production_player: str,
    ) -> tuple[list, int, set[HexCoord]]:
        actions: list = []
        counts = self._unit_counts(world)
        unit_cap = self._unit_cap(world)
        pending_count = len(self._pending_production)
        gold_reserve = self._expansion_gold_reserve(world, production_player)
        if sum(counts.values()) + pending_count >= unit_cap:
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
            remaining_capacity = unit_cap - sum(counts.values()) - pending_count
            if remaining_capacity <= 0:
                break
            slots = self._spawn_slots(world, building, reserved)
            if not slots:
                continue
            slots = slots[:remaining_capacity]
            wants = self._production_wants(
                world, building, counts, len(slots), strategy, production_player
            )
            for unit_type in wants:
                if len(actions) >= _PRODUCE_LIMIT_PER_TURN or not slots:
                    break
                cost = UNIT_STATS[unit_type].gold_cost
                if gold < cost or gold - cost < gold_reserve:
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

    def _expansion_gold_reserve(self, world: World, production_player: str) -> int:
        threatened = self._base_threatened(world)
        if threatened and production_player == "defense":
            return 0

        bases = [b for b in world.own_buildings if b["type"] == "Base"]
        desired_bases = 1 + (world.turn >= 28) + (world.turn >= 85) + (world.turn >= 145)
        if threatened:
            desired_bases += 1
        desired_bases = min(desired_bases, 4)

        if len(bases) >= desired_bases:
            return 0
        if world.turn < 28:
            return 0
        return BUILDING_STATS["Base"].gold_cost

    def _production_wants(
        self,
        world: World,
        building: dict,
        counts: dict[str, int],
        slots: int,
        strategy: StrategyDirective,
        production_player: str,
    ) -> list[str]:
        btype = building["type"]
        total_units = sum(counts.values())
        threatened = self._base_threatened(world)
        enemy_bomber_seen = any(e["type"] == "Bomber" for e in world.enemies)
        enemy_base_seen = any(e["type"] == "Base" for e in world.enemies)
        finishable_base = self._finishable_enemy_base(world)
        wants: list[str] = []

        if btype == "Barracks":
            desired_scouts = 2 + (world.turn >= 90) + (
                strategy.macro_directive == "SCOUT_EXPANSION"
            )
            if counts["Scout"] < desired_scouts:
                wants.append("Scout")
            if (
                counts["Medic"] < max(1, counts["Infantry"] // 5)
                and total_units >= 4
            ) or production_player == "defense":
                wants.append("Medic")
            wants.extend(["Infantry"] * max(1, slots - len(wants)))
            if threatened:
                wants.extend(["Infantry"] * 3)
        elif btype == "Factory":
            if finishable_base is not None:
                wants.extend(["Artillery", "Tank", "Artillery"])
            if production_player in ("defense", "ground_assault") or threatened:
                wants.append("Tank")
            if strategy.macro_directive.startswith("BASE_ASSAULT"):
                wants.extend(["Artillery", "Tank"])
            elif counts["Tank"] <= counts["Artillery"]:
                wants.append("Tank")
            wants.append("Artillery")
            wants.extend(["Tank", "Artillery"] * max(0, slots))
        elif btype == "Airbase":
            if (
                enemy_bomber_seen
                or production_player in ("anti_air", "air_rush")
                or counts["Fighter"] < counts["Bomber"] + 2
            ):
                wants.append("Fighter")
            if (
                strategy.macro_directive in ("BASE_ASSAULT_AIR", "BASE_ASSAULT")
                and (enemy_base_seen or finishable_base is not None or world.turn > 120)
            ):
                wants.append("Bomber")
            if finishable_base is not None:
                wants.extend(["Bomber", "Fighter"])
            if production_player == "air_rush":
                wants.extend(["Fighter", "Bomber"])
            wants.extend(["Fighter"] * max(0, slots - len(wants)))

        return wants[: max(1, min(slots, 4))]

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

    # -- combat and movement ----------------------------------------------

    def _combat_and_moves(
        self,
        world: World,
        reserved: set[HexCoord],
        influence: InfluenceMaps,
        strategy: StrategyDirective,
        movement_player: str,
        combat_player: str,
    ) -> tuple[list, set[HexCoord]]:
        actions: list = []
        move_reserved = set(reserved)
        own_coords = {self._coord(e) for e in world.own_units + world.own_buildings}
        incoming_damage: dict[str, int] = defaultdict(int)

        for unit in sorted(
            world.own_units, key=lambda u: (u["type"], u["r"], u["q"], u["id"])
        ):
            here = self._coord(unit)
            target = self._best_attack_coord(
                world, unit, incoming_damage, own_coords, combat_player, strategy
            )
            if target is not None:
                actions.append(AttackAction(unit_id=unit["id"], target=target))
                direct = self._entity_at(world.enemies, target)
                if direct:
                    incoming_damage[direct["id"]] += self._attack_power(world, unit, direct)

            if not world.enemies and unit.get("type") != "Scout":
                continue

            dest = self._best_move(
                world, unit, move_reserved, influence, strategy, movement_player
            )
            if dest is not None and dest != here:
                path = self._movement_path(world, here, dest, int(unit["movement_range"]), move_reserved)
                if path and len(path) > 1:
                    actions.append(MoveAction(unit_id=unit["id"], path=path))
                    move_reserved.discard(here)
                    move_reserved.add(path[-1])

        return actions, move_reserved

    def _best_attack_coord(
        self,
        world: World,
        unit: dict,
        incoming_damage: dict[str, int],
        own_coords: set[HexCoord],
        combat_player: str,
        strategy: StrategyDirective,
    ) -> HexCoord | None:
        attack_range = int(unit.get("attack_range", 0))
        if attack_range <= 0:
            return None
        here = self._coord(unit)

        if unit["type"] == "Artillery":
            artillery_target = self._best_artillery_target(
                world, unit, own_coords, combat_player, strategy
            )
            if artillery_target is not None:
                return artillery_target

        candidates: list[tuple[tuple, HexCoord]] = []
        for enemy in world.enemies:
            target = self._coord(enemy)
            dist = world.grid.distance(here, target)
            if not (0 < dist <= attack_range):
                continue
            hp_left = int(enemy.get("hp", 0)) - incoming_damage[enemy["id"]]
            score = self._target_score(world, unit, enemy, dist, hp_left, combat_player, strategy)
            candidates.append((score, target))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    def _best_artillery_target(
        self,
        world: World,
        unit: dict,
        own_coords: set[HexCoord],
        combat_player: str,
        strategy: StrategyDirective,
    ) -> HexCoord | None:
        here = self._coord(unit)
        attack_range = int(unit.get("attack_range", 0))
        candidate_coords: set[HexCoord] = set()
        for enemy in world.enemies:
            ec = self._coord(enemy)
            if 0 < world.grid.distance(here, ec) <= attack_range:
                candidate_coords.add(ec)
            for nb in world.grid.neighbors(ec):
                if 0 < world.grid.distance(here, nb) <= attack_range:
                    candidate_coords.add(nb)

        scored: list[tuple[tuple, HexCoord]] = []
        for coord in candidate_coords:
            primary = self._entity_at(world.own_units + world.own_buildings, coord)
            if primary is not None:
                continue
            value = 0.0
            direct = self._entity_at(world.enemies, coord)
            if direct:
                value += self._attack_value(world, unit, direct, strategy)
            for splash_coord in world.grid.ring(coord, 1):
                enemy = self._entity_at(world.enemies, splash_coord)
                if enemy:
                    value += self._attack_value(world, unit, enemy, strategy) * ARTILLERY_SPLASH_DAMAGE_RATIO
                if splash_coord in own_coords:
                    value -= 35.0
            if combat_player == "artillery_cluster":
                value *= 1.25
            if value <= 0:
                continue
            scored.append(((-value, coord.r, coord.q), coord))
        if not scored:
            return None
        return min(scored, key=lambda item: item[0])[1]

    def _target_score(
        self,
        world: World,
        unit: dict,
        target: dict,
        dist: int,
        hp_left: int,
        combat_player: str,
        strategy: StrategyDirective,
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
            "Airbase": 2,
            "Factory": 3,
            "Tank": 4,
            "Fighter": 5,
            "Base": 6,
            "Infantry": 7,
            "Scout": 8,
            "Medic": 9,
            "Barracks": 10,
            "Mine": 11,
        }.get(ttype, 12)
        if combat_player == "anti_bomber" and ttype in ("Bomber", "Airbase"):
            type_priority -= 5
        if combat_player == "base_snipe" and ttype in ("Base", "Airbase", "Factory"):
            type_priority -= 6
        if strategy.macro_directive.startswith("BASE_ASSAULT") and ttype in BUILDING_STATS:
            type_priority -= 3
        if unit["type"] == "Bomber" and ttype in BUILDING_STATS:
            type_priority -= 4
        killable = 0 if hp_left <= self._attack_power(world, unit, target) else 1
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
        self,
        world: World,
        unit: dict,
        reserved: set[HexCoord],
        influence: InfluenceMaps,
        strategy: StrategyDirective,
        movement_player: str,
    ) -> HexCoord | None:
        here = self._coord(unit)
        movement = int(unit.get("movement_range", 0))
        if movement <= 0:
            return None
        reachable = self._reachable_this_turn(world, here, movement, reserved)
        if not reachable:
            return None

        if movement_player == "medic_pulse":
            heal_dest = self._best_heal_move(world, unit, here, reachable, influence)
            if heal_dest is not None:
                return heal_dest

        if unit["type"] == "Scout":
            return self._best_scout_move(world, here, reachable, influence, movement_player)
        if unit["type"] == "Fighter":
            return self._best_fighter_move(world, here, reachable, influence, movement_player)
        if unit["type"] == "Bomber":
            bomber_dest = self._best_bomber_move(world, here, reachable, influence, strategy)
            if bomber_dest is not None:
                return bomber_dest

        enemies = world.enemies
        bases = [self._coord(b) for b in world.own_buildings if b["type"] == "Base"]
        if enemies:
            attack_range = int(unit.get("attack_range", 0))
            targets = self._movement_targets(world, strategy, movement_player)
            own_ground = [
                self._coord(u)
                for u in world.own_units
                if u["id"] != unit["id"] and u.get("type") in ("Infantry", "Tank")
            ]

            def score(coord: HexCoord) -> tuple:
                nearest_target = min(world.grid.distance(coord, t) for t in targets)
                nearest_base = min(
                    (world.grid.distance(coord, b) for b in bases), default=0
                )
                in_range = 0 if attack_range and nearest_target <= attack_range else 1
                defend = 0 if movement_player in ("guard", "intercept") and nearest_base <= 4 else 1
                terrain_score = 0 if self._terrain(world, coord) == "elevated" else 1
                threat = influence.value("threat", coord)
                protection = influence.value("protection", coord)
                cluster = sum(1 for oc in own_ground if world.grid.distance(coord, oc) <= 1)
                if movement_player == "assault":
                    return (in_range, nearest_target, threat, cluster, terrain_score, coord.r, coord.q)
                return (
                    defend,
                    in_range,
                    nearest_target,
                    threat,
                    -protection,
                    cluster,
                    terrain_score,
                    coord.r,
                    coord.q,
                )

            return min(reachable, key=score)

        if bases:
            def guard_score(coord: HexCoord) -> tuple:
                nearest_base = min(world.grid.distance(coord, b) for b in bases)
                terrain_score = 0 if self._terrain(world, coord) == "elevated" else 1
                ideal_ring = abs(nearest_base - 2)
                threat = influence.value("threat", coord)
                protection = influence.value("protection", coord)
                return (ideal_ring, threat, -protection, terrain_score, nearest_base, coord.r, coord.q)

            best = min(reachable, key=guard_score)
            if guard_score(best) < guard_score(here):
                return best
        return None

    def _best_heal_move(
        self,
        world: World,
        unit: dict,
        here: HexCoord,
        reachable: set[HexCoord],
        influence: InfluenceMaps,
    ) -> HexCoord | None:
        medics = [self._coord(u) for u in world.own_units if u.get("type") == "Medic"]
        if not medics or unit.get("type") not in ("Infantry", "Tank", "Artillery"):
            return None
        if self._hp_ratio(unit) > 0.68:
            return None

        def score(coord: HexCoord) -> tuple:
            medic_dist = min(world.grid.distance(coord, medic) for medic in medics)
            threat = influence.value("threat", coord)
            protection = influence.value("protection", coord)
            return (medic_dist, threat, -protection, coord.r, coord.q)

        best = min(reachable, key=score)
        return best if score(best) < score(here) else None

    def _best_scout_move(
        self,
        world: World,
        here: HexCoord,
        reachable: set[HexCoord],
        influence: InfluenceMaps,
        movement_player: str,
    ) -> HexCoord | None:
        own_bases = [self._coord(b) for b in world.own_buildings if b["type"] == "Base"]
        enemy_coords = [self._coord(e) for e in world.enemies]
        seen = self._seen_terrain
        scan_radius = 3 if len(world.own_units) >= 8 else 4

        def score(coord: HexCoord) -> tuple:
            newly_seen = 0
            rich_seen = 0
            for c in self._disk(world, coord, scan_radius):
                if c not in seen:
                    newly_seen += 1
                if self._terrain(world, c) == "rich_resource":
                    rich_seen += 1
            enemy_dist = min(
                (self._distance(world, coord, e) for e in enemy_coords), default=99
            )
            base_dist = min(
                (self._distance(world, coord, b) for b in own_bases), default=0
            )
            conceal = influence.value("concealment", coord)
            threat = influence.value("threat", coord)
            danger = 0 if enemy_dist > 2 else 1
            scout_bonus = -conceal * (4 if movement_player == "scout" else 2)
            return (-newly_seen, -rich_seen, danger, threat, scout_bonus, -base_dist, coord.r, coord.q)

        best = min(reachable, key=score)
        return best if score(best) < score(here) else None

    def _best_fighter_move(
        self,
        world: World,
        here: HexCoord,
        reachable: set[HexCoord],
        influence: InfluenceMaps,
        movement_player: str,
    ) -> HexCoord | None:
        air_threats = [
            self._coord(e)
            for e in world.enemies
            if e.get("type") in ("Bomber", "Fighter", "Airbase")
        ]
        bases = [self._coord(b) for b in world.own_buildings if b.get("type") == "Base"]
        if not air_threats and movement_player != "intercept":
            return None

        def score(coord: HexCoord) -> tuple:
            threat_dist = min((world.grid.distance(coord, t) for t in air_threats), default=99)
            base_dist = min((world.grid.distance(coord, b) for b in bases), default=0)
            protection = influence.value("protection", coord)
            threat = influence.value("threat", coord)
            return (threat_dist, base_dist, threat, -protection, coord.r, coord.q)

        best = min(reachable, key=score)
        return best if score(best) < score(here) else None

    def _best_bomber_move(
        self,
        world: World,
        here: HexCoord,
        reachable: set[HexCoord],
        influence: InfluenceMaps,
        strategy: StrategyDirective,
    ) -> HexCoord | None:
        targets = [
            self._coord(e)
            for e in world.enemies
            if e.get("type") in ("Base", "Airbase", "Factory", "Barracks", "Mine")
        ]
        if not targets:
            return None

        def score(coord: HexCoord) -> tuple:
            target_dist = min(world.grid.distance(coord, t) for t in targets)
            threat = influence.value("threat", coord)
            return (target_dist, threat * (1.0 - strategy.risk_tolerance), coord.r, coord.q)

        return min(reachable, key=score)

    def _movement_targets(
        self, world: World, strategy: StrategyDirective, movement_player: str
    ) -> list[HexCoord]:
        priority = set(strategy.priority_targets)
        finishable_base = self._finishable_enemy_base(world)
        if finishable_base is not None:
            return [self._coord(finishable_base)]
        if movement_player == "intercept":
            targets = [
                self._coord(e)
                for e in world.enemies
                if e.get("type") in ("Bomber", "Fighter", "Airbase")
            ]
            if targets:
                return targets
        if movement_player == "assault" or strategy.macro_directive.startswith("BASE_ASSAULT"):
            targets = [
                self._coord(e)
                for e in world.enemies
                if e.get("type") in ("Base", "Airbase", "Factory")
                or e.get("type") in priority
                or e.get("id") in priority
            ]
            if targets:
                return targets
        targets = [self._coord(e) for e in world.enemies if e.get("type") in priority or e.get("id") in priority]
        if targets:
            return targets
        return [self._coord(e) for e in world.enemies]

    def _reachable_this_turn(
        self,
        world: World,
        origin: HexCoord,
        movement: int,
        reserved: set[HexCoord],
    ) -> set[HexCoord]:
        blocked = {c for c in reserved if c != origin}
        costs = self._movement_costs(world)
        return world.grid.reachable(origin, movement, costs, blocked)

    def _movement_path(
        self,
        world: World,
        origin: HexCoord,
        target: HexCoord,
        movement: int,
        reserved: set[HexCoord],
    ) -> list[HexCoord] | None:
        blocked = {c for c in reserved if c != origin and c != target}
        path = world.grid.shortest_path(origin, target, self._movement_costs(world), blocked)
        if not path:
            return None
        cost = sum(self._movement_costs(world).get(c, 1) for c in path[1:])
        if cost > movement or len(path) - 1 > movement:
            return None
        return path

    def _movement_costs(self, world: World) -> dict[HexCoord, int]:
        if self._movement_cost_cache is None:
            self._movement_cost_cache = {
                c: DIFFICULT_TERRAIN_MOVE_COST
                for c, terrain in {**self._seen_terrain, **world.terrain}.items()
                if terrain == "difficult"
            }
        return self._movement_cost_cache

    def _distance(self, world: World, a: HexCoord, b: HexCoord) -> int:
        aw = world.grid.wrap(a)
        bw = world.grid.wrap(b)
        left = (aw.q, aw.r)
        right = (bw.q, bw.r)
        if right < left:
            left, right = right, left
        key = (world.grid.width, world.grid.height, left[0], left[1], right[0], right[1])
        cached = self._distance_cache.get(key)
        if cached is None:
            cached = world.grid.distance(aw, bw)
            self._distance_cache[key] = cached
        return cached

    def _disk(self, world: World, coord: HexCoord, radius: int) -> tuple[HexCoord, ...]:
        wrapped = world.grid.wrap(coord)
        key = (world.grid.width, world.grid.height, wrapped.q, wrapped.r, radius)
        cached = self._disk_cache.get(key)
        if cached is None:
            cached = tuple(world.grid.disk(wrapped, radius))
            self._disk_cache[key] = cached
        return cached

    # -- values and utilities ---------------------------------------------

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

    def _reconcile_pending_production(self, world: World) -> None:
        live_buildings = {
            b["id"]
            for b in world.own_buildings
            if b.get("type") in _PRODUCTION_BUILDINGS and b.get("is_complete", True)
        }
        self._pending_production = [
            pending
            for pending in self._pending_production
            if pending.due_turn >= world.turn and pending.building_id in live_buildings
        ]

    def _pending_spawn_reservations(self, world: World) -> set[HexCoord]:
        building_by_id = {b["id"]: b for b in world.own_buildings}
        reserved: set[HexCoord] = set()
        for pending in self._pending_production:
            if pending.due_turn > world.turn:
                continue
            building = building_by_id.get(pending.building_id)
            if building is None:
                continue
            building_coord = self._coord(building)
            target = world.grid.wrap(pending.target)
            candidates = [target] + world.grid.neighbors(building_coord)
            added = 0
            for coord in candidates:
                if coord in world.occupied or coord in reserved:
                    continue
                reserved.add(coord)
                added += 1
                if added >= 2:
                    break
        return reserved

    def _remember_production_orders(self, world: World, actions: list) -> None:
        for action in actions:
            if not isinstance(action, ProduceUnitAction):
                continue
            stats = UNIT_STATS.get(action.unit_type)
            if stats is None:
                continue
            self._pending_production.append(
                PendingProduction(
                    building_id=action.building_id,
                    unit_type=action.unit_type,
                    target=world.grid.wrap(action.target),
                    due_turn=world.turn + stats.build_turns,
                )
            )

    def _recent_enemy_types(self, world: World, max_age: int) -> set[str]:
        return {
            entity_type
            for _, seen_turn, entity_type in self._last_seen_enemy.values()
            if entity_type and 0 <= world.turn - seen_turn <= max_age
        }

    def _imminent_hostile_players(self, world: World) -> set[str]:
        base_coords = [
            self._coord(base)
            for base in world.own_buildings
            if base.get("type") == "Base" and base.get("is_complete", True)
        ]
        hostiles: set[str] = set()
        for enemy in world.enemies:
            owner = enemy.get("owner_id")
            if not owner:
                continue
            enemy_type = enemy.get("type")
            if enemy_type in ("Bomber", "Airbase", "Factory", "Artillery"):
                hostiles.add(owner)
                continue
            if base_coords:
                ec = self._coord(enemy)
                if any(world.grid.distance(ec, bc) <= 7 for bc in base_coords):
                    hostiles.add(owner)
        return hostiles

    def _finishable_enemy_base(self, world: World) -> dict | None:
        cached = self._turn_cache.get("finishable_enemy_base")
        if cached is not None:
            return cached if isinstance(cached, dict) else None

        bases = [enemy for enemy in world.enemies if enemy.get("type") == "Base"]
        if not bases:
            self._turn_cache["finishable_enemy_base"] = False
            return None
        counts = self._unit_counts(world)
        attack_mass = (
            counts["Bomber"] * 200
            + counts["Artillery"] * 75
            + counts["Tank"] * 60
            + counts["Fighter"] * 35
            + counts["Infantry"] * 20
        )

        def score(base: dict) -> tuple:
            hp = int(base.get("hp", 999))
            hp_ratio = self._hp_ratio(base)
            base_coord = self._coord(base)
            own_dist = min(
                (
                    world.grid.distance(base_coord, self._coord(unit))
                    for unit in world.own_units
                ),
                default=99,
            )
            return (hp_ratio, hp, own_dist, base_coord.r, base_coord.q)

        best = min(bases, key=score)
        best_hp = int(best.get("hp", 999))
        if best_hp <= 160 or self._hp_ratio(best) <= 0.55 or attack_mass >= best_hp * 1.4:
            self._turn_cache["finishable_enemy_base"] = best
            return best
        self._turn_cache["finishable_enemy_base"] = False
        return None

    def _base_threatened(self, world: World) -> bool:
        cached = self._turn_cache.get("base_threatened")
        if isinstance(cached, bool):
            return cached

        bases = [b for b in world.own_buildings if b["type"] == "Base"]
        if any(int(b.get("hp", 999)) < int(b.get("max_hp", 999)) for b in bases):
            self._turn_cache["base_threatened"] = True
            return True
        base_coords = [self._coord(b) for b in bases]
        for enemy in world.enemies:
            ec = self._coord(enemy)
            threat_radius = 8 if enemy.get("type") in ("Bomber", "Artillery") else 6
            if any(world.grid.distance(ec, bc) <= threat_radius for bc in base_coords):
                self._turn_cache["base_threatened"] = True
                return True
        for coord, seen_turn, entity_type in self._last_seen_enemy.values():
            if world.turn - seen_turn > 10:
                continue
            threat_radius = 9 if entity_type in ("Bomber", "Artillery") else 6
            if entity_type in ("Bomber", "Artillery", "Tank", "Airbase", "Factory") and any(
                world.grid.distance(coord, bc) <= threat_radius for bc in base_coords
            ):
                self._turn_cache["base_threatened"] = True
                return True
        self._turn_cache["base_threatened"] = False
        return False

    def _unit_cap(self, world: World) -> int:
        cap = min(62, 10 + world.turn // 4)
        if self._base_threatened(world):
            cap += 10
        return cap

    @staticmethod
    def _hp_ratio(entity: dict) -> float:
        max_hp = max(1, int(entity.get("max_hp", 1)))
        return max(0.0, min(1.0, int(entity.get("hp", max_hp)) / max_hp))

    @staticmethod
    def _unit_counts(world: World) -> defaultdict[str, int]:
        counts: defaultdict[str, int] = defaultdict(int)
        for unit in world.own_units:
            counts[unit["type"]] += 1
        return counts

    @staticmethod
    def _entity_at(entities: list[dict], coord: HexCoord) -> dict | None:
        for entity in entities:
            if int(entity["q"]) == coord.q and int(entity["r"]) == coord.r:
                return entity
        return None

    @staticmethod
    def _own_unit_by_id(world: World, unit_id: str) -> dict | None:
        for unit in world.own_units:
            if unit["id"] == unit_id:
                return unit
        return None

    def _near_own_base(self, world: World, coord: HexCoord, radius: int) -> bool:
        return any(
            world.grid.distance(coord, self._coord(base)) <= radius
            for base in world.own_buildings
            if base.get("type") == "Base"
        )

    def _attack_power(self, world: World, attacker: dict, target: dict) -> int:
        power = int(attacker.get("attack_power", 0))
        if self._terrain(world, self._coord(attacker)) == "elevated":
            power = int(power * ELEVATION_ATTACK_BONUS)
        if attacker.get("type") == "Bomber" and target.get("type") in BUILDING_STATS:
            power = int(power * Bomber.BUILDING_DAMAGE_MULTIPLIER)
        return power

    def _attack_value(
        self,
        world: World,
        attacker: dict | None,
        target: dict,
        strategy: StrategyDirective,
    ) -> float:
        if attacker is None:
            damage = 30
        else:
            damage = self._attack_power(world, attacker, target)
        hp = max(1, int(target.get("hp", 1)))
        type_value = {
            "Base": 140,
            "Bomber": 80,
            "Airbase": 70,
            "Factory": 58,
            "Artillery": 52,
            "Fighter": 48,
            "Tank": 42,
            "Barracks": 34,
            "Mine": 30,
            "Infantry": 24,
            "Scout": 22,
            "Medic": 18,
        }.get(target.get("type", ""), 12)
        if target.get("type") in strategy.priority_targets or target.get("id") in strategy.priority_targets:
            type_value *= 1.35
        if strategy.macro_directive.startswith("BASE_ASSAULT") and target.get("type") in BUILDING_STATS:
            type_value *= 1.25
        kill_bonus = 35.0 if damage >= hp else 0.0
        return type_value * min(1.0, damage / hp) + kill_bonus

    @staticmethod
    def _building_value(building_type: str, strategy: StrategyDirective) -> float:
        base = {
            "Base": 65.0,
            "Mine": 40.0,
            "Barracks": 34.0,
            "Factory": 46.0,
            "Airbase": 55.0,
        }.get(building_type, 10.0)
        weights = strategy.resource_weights
        if building_type in _RESOURCE_BUILDINGS:
            base *= 1.0 + weights.get("economy", 0.0)
        if building_type in ("Barracks", "Factory"):
            base *= 1.0 + weights.get("ground", 0.0)
        if building_type == "Airbase":
            base *= 1.0 + weights.get("air", 0.0)
        if building_type == "Base":
            base *= 1.0 + weights.get("defense", 0.0)
        return base

    @staticmethod
    def _unit_value(unit_type: str, strategy: StrategyDirective) -> float:
        base = {
            "Infantry": 18.0,
            "Scout": 26.0,
            "Medic": 24.0,
            "Tank": 45.0,
            "Artillery": 48.0,
            "Fighter": 60.0,
            "Bomber": 66.0,
        }.get(unit_type, 10.0)
        weights = strategy.resource_weights
        if unit_type in _AIR_TYPES:
            base *= 1.0 + weights.get("air", 0.0)
        if unit_type in _GROUND_TYPES:
            base *= 1.0 + weights.get("ground", 0.0)
        if unit_type in ("Infantry", "Tank", "Fighter"):
            base *= 1.0 + 0.4 * weights.get("defense", 0.0)
        if unit_type in ("Scout", "Medic"):
            base *= 1.0 + 0.25 * weights.get("economy", 0.0)
        return base

    def _movement_value(
        self, world: World, strategy: StrategyDirective, unit: dict, dest: HexCoord
    ) -> float:
        value = 0.0
        if world.enemies:
            targets = self._movement_targets(world, strategy, "assault")
            nearest = min(world.grid.distance(dest, target) for target in targets)
            value += max(0, 12 - nearest) * strategy.aggression
        if unit.get("type") == "Scout":
            value += sum(
                0.35 for coord in world.grid.disk(dest, 5) if coord not in self._seen_terrain
            )
        if unit.get("type") in ("Infantry", "Tank") and self._hp_ratio(unit) < 0.7:
            medics = [self._coord(u) for u in world.own_units if u.get("type") == "Medic"]
            if medics:
                nearest_medic = min(world.grid.distance(dest, medic) for medic in medics)
                value += max(0, 3 - nearest_medic) * 3.0
        return value


# Keep the entity imports live and explicit for the PRD's engine-contract check.
_ENGINE_ENTITY_TYPES = (
    Base,
    Mine,
    Barracks,
    Factory,
    Airbase,
    Infantry,
    Tank,
    Artillery,
    Scout,
    Medic,
    Fighter,
    Bomber,
)
