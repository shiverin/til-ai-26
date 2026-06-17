"""Hybrid LLM strategy manager.

The LLM is used only for macroscopic GOAM directives.  It never emits raw moves
or coordinates; the deterministic ``AlgoAgent`` owns all action construction,
hex math, HPS/MAPLE evaluation, and engine-rule compliance.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

from agent_base import PlayerAgent
from algo_agent import AlgoAgent, StrategyDirective
from engine.actions import ActionPayload
from engine.constants import BUILDING_STATS, RICH_RESOURCE_FLAT_YIELD, UNIT_STATS
from engine.hex_grid import HexCoord, HexGrid
from llm import call_llm, parse_json


_REFRESH_INTERVAL_TURNS = 10
_THREAT_REFRESH_INTERVAL_TURNS = 3
_STRATEGY_TIMEOUT_SECONDS = 2.8

_SYSTEM = """\
You are the Strategy Manager for one team in a 20-player turn-based hex-grid wargame.
You receive compressed state summaries generated from engine Entity.to_dict() payloads.

Return ONLY a JSON object with this exact schema:
{
  "macro_directive": "BALANCED|ECONOMIC_EXPANSION|SCOUT_EXPANSION|DEFENSIVE_TURTLE|AIR_SUPERIORITY_RUSH|BASE_ASSAULT|BASE_ASSAULT_AIR",
  "resource_weights": {"air": 0.0, "ground": 0.0, "economy": 0.0, "defense": 0.0},
  "priority_targets": ["Base|Bomber|Airbase|Factory|Artillery|Tank|Scout"],
  "aggression": 0.0,
  "risk_tolerance": 0.0
}

Do not emit concrete game actions, unit ids, paths, or coordinates. The local
algorithmic managers handle all production, pathfinding, targeting, HPS, and
MAPLE search. Choose directives only from the enum above.
"""


class GameStateSerializer:
    """Compress the observation into macro metrics for the remote LLM.

    The observation's entity dictionaries are the wire-format output of the
    engine's Entity.to_dict() methods, so this serializer aggregates them rather
    than asking the LLM to reason over raw visible-tile arrays.
    """

    def serialize(self, observation: dict) -> dict:
        pid = observation["player_id"]
        grid = HexGrid(
            int(observation.get("map_width", 35)),
            int(observation.get("map_height", 30)),
        )
        friendly_counts: Counter[str] = Counter()
        enemy_counts: Counter[str] = Counter()
        friendly_base_hp: list[dict] = []
        enemy_base_hp: list[dict] = []
        enemy_bombers: list[dict] = []
        enemy_airbases: list[dict] = []
        visible_enemy_bases: list[dict] = []
        zones: dict[str, dict[str, int]] = defaultdict(
            lambda: {
                "friendly_entities": 0,
                "enemy_entities": 0,
                "friendly_power": 0,
                "enemy_power": 0,
                "rich_tiles": 0,
                "concealment_tiles": 0,
                "visible_tiles": 0,
            }
        )
        income = {"friendly": 0, "visible_enemy": 0}
        active_mines = {"friendly": 0, "visible_enemy": 0}
        terrain_counts: Counter[str] = Counter()

        for tile in observation.get("visible_tiles", []):
            coord = grid.wrap(HexCoord(int(tile["q"]), int(tile["r"])))
            zone = self._zone_name(grid, coord)
            terrain = tile.get("terrain", "normal")
            terrain_counts[terrain] += 1
            zones[zone]["visible_tiles"] += 1
            if terrain == "rich_resource":
                zones[zone]["rich_tiles"] += 1
            if terrain == "concealment":
                zones[zone]["concealment_tiles"] += 1

            for entity in tile.get("entities", []):
                owner = entity.get("owner_id")
                etype = entity.get("type", "")
                is_friendly = owner == pid
                target_counts = friendly_counts if is_friendly else enemy_counts
                target_counts[etype] += 1

                power = self._entity_power(entity)
                if is_friendly:
                    zones[zone]["friendly_entities"] += 1
                    zones[zone]["friendly_power"] += power
                else:
                    zones[zone]["enemy_entities"] += 1
                    zones[zone]["enemy_power"] += power

                if etype in ("Base", "Mine") and entity.get("is_complete", True):
                    amount = self._building_income(etype, terrain)
                    side = "friendly" if is_friendly else "visible_enemy"
                    income[side] += amount
                    if etype == "Mine":
                        active_mines[side] += 1

                if etype == "Base":
                    base_record = {
                        "owner_id": owner,
                        "hp": int(entity.get("hp", 0)),
                        "max_hp": int(entity.get("max_hp", 0)),
                        "zone": zone,
                    }
                    if is_friendly:
                        friendly_base_hp.append(base_record)
                    else:
                        enemy_base_hp.append(base_record)
                        visible_enemy_bases.append(base_record)
                elif not is_friendly and etype == "Bomber":
                    enemy_bombers.append(
                        {
                            "owner_id": owner,
                            "hp": int(entity.get("hp", 0)),
                            "zone": zone,
                        }
                    )
                elif not is_friendly and etype == "Airbase":
                    enemy_airbases.append(
                        {
                            "owner_id": owner,
                            "hp": int(entity.get("hp", 0)),
                            "complete": bool(entity.get("is_complete", True)),
                            "zone": zone,
                        }
                    )

        friendly_air = sum(friendly_counts[t] for t in ("Fighter", "Bomber"))
        enemy_air = sum(enemy_counts[t] for t in ("Fighter", "Bomber"))
        friendly_ground = sum(friendly_counts[t] for t in UNIT_STATS if t not in ("Fighter", "Bomber"))
        enemy_ground = sum(enemy_counts[t] for t in UNIT_STATS if t not in ("Fighter", "Bomber"))

        return {
            "turn": int(observation.get("turn_number", 0)),
            "max_turns": int(observation.get("max_turns", 300)),
            "player_id": pid,
            "gold": int(observation.get("resources", {}).get("gold", 0)),
            "map": {
                "width": grid.width,
                "height": grid.height,
                "visible_tiles": len(observation.get("visible_tiles", [])),
                "terrain_counts": dict(terrain_counts),
            },
            "base_hp": {
                "friendly": friendly_base_hp,
                "visible_enemy": enemy_base_hp,
            },
            "economy": {
                "estimated_income": income,
                "active_mines": active_mines,
                "income_delta": income["friendly"] - income["visible_enemy"],
                "mine_delta": active_mines["friendly"] - active_mines["visible_enemy"],
            },
            "force_mix": {
                "friendly_counts": dict(friendly_counts),
                "visible_enemy_counts": dict(enemy_counts),
                "friendly_air": friendly_air,
                "visible_enemy_air": enemy_air,
                "friendly_ground": friendly_ground,
                "visible_enemy_ground": enemy_ground,
            },
            "asymmetrical_threats": {
                "enemy_bombers": enemy_bombers,
                "enemy_bomber_count": len(enemy_bombers),
                "enemy_airbases": enemy_airbases,
                "friendly_base_damaged": any(
                    b["max_hp"] and b["hp"] < b["max_hp"] for b in friendly_base_hp
                ),
                "visible_enemy_bases": visible_enemy_bases,
            },
            "zones": dict(zones),
            "diplomacy": {
                "known_players": list(observation.get("known_players", []))[:20],
                "treaties": observation.get("treaties", [])[:20],
                "incoming_proposals": observation.get("incoming_treaty_proposals", [])[:20],
            },
        }

    @staticmethod
    def _building_income(entity_type: str, terrain: str) -> int:
        if terrain == "rich_resource":
            return RICH_RESOURCE_FLAT_YIELD
        if entity_type not in BUILDING_STATS:
            return 0
        return BUILDING_STATS[entity_type].gold_yield_per_turn

    @staticmethod
    def _entity_power(entity: dict) -> int:
        etype = entity.get("type", "")
        hp = int(entity.get("hp", 0))
        if etype in UNIT_STATS:
            stats = UNIT_STATS[etype]
            return hp + stats.attack_power * max(1, stats.attack_range)
        if etype in BUILDING_STATS:
            return hp // 2
        return hp

    @staticmethod
    def _zone_name(grid: HexGrid, coord: HexCoord) -> str:
        wrapped = grid.wrap(coord)
        col = (wrapped.q + wrapped.r // 2) % grid.width
        vertical = (
            "north" if wrapped.r < grid.height / 3 else "south" if wrapped.r >= 2 * grid.height / 3 else "central"
        )
        horizontal = (
            "west" if col < grid.width / 3 else "east" if col >= 2 * grid.width / 3 else "mid"
        )
        return f"{vertical}_{horizontal}"


class LLMAgent(PlayerAgent):
    def __init__(self) -> None:
        self._serializer = GameStateSerializer()
        self._algo = AlgoAgent()
        self._last_strategy_turn = -10_000
        self._last_strategy_attempt_turn = -10_000
        self._cached_strategy = StrategyDirective()
        self._has_llm_strategy = False

    async def decide(self, observation: dict) -> ActionPayload:
        if self._should_refresh_strategy(observation):
            self._last_strategy_attempt_turn = int(observation.get("turn_number", 0))
            strategy = await self._query_strategy(observation)
            if strategy is not None:
                self._cached_strategy = strategy
                self._last_strategy_turn = int(observation.get("turn_number", 0))
                self._has_llm_strategy = True

        directive = self._cached_strategy if self._has_llm_strategy else None
        return await self._algo.decide(observation, directive)

    def _should_refresh_strategy(self, observation: dict) -> bool:
        if not os.environ.get("OPENROUTER_API_KEY"):
            return False
        turn = int(observation.get("turn_number", 0))
        if turn - self._last_strategy_attempt_turn < _THREAT_REFRESH_INTERVAL_TURNS:
            return False
        if turn == 0:
            return True
        if turn - self._last_strategy_turn >= _REFRESH_INTERVAL_TURNS:
            return True
        summary = self._serializer.serialize(observation)
        threats = summary["asymmetrical_threats"]
        return bool(
            threats["enemy_bomber_count"] or threats["friendly_base_damaged"]
        ) and turn - self._last_strategy_turn >= _THREAT_REFRESH_INTERVAL_TURNS

    async def _query_strategy(self, observation: dict) -> StrategyDirective | None:
        summary = self._serializer.serialize(observation)
        user = json.dumps(summary, separators=(",", ":"), sort_keys=True)
        reply = await call_llm(
            _SYSTEM,
            user,
            max_tokens=260,
            timeout=_STRATEGY_TIMEOUT_SECONDS,
            temperature=0.0,
        )
        data = parse_json(reply)
        if not data:
            return None
        return self._guard_strategy(summary, StrategyDirective.from_payload(data))

    @staticmethod
    def _guard_strategy(summary: dict, strategy: StrategyDirective) -> StrategyDirective:
        threats = summary.get("asymmetrical_threats", {})
        force_mix = summary.get("force_mix", {})
        enemy_bases = threats.get("visible_enemy_bases") or []
        low_enemy_base = min(
            (
                int(base.get("hp", 999))
                for base in enemy_bases
                if int(base.get("max_hp", 0) or 0) > 0
            ),
            default=999,
        )

        if threats.get("friendly_base_damaged"):
            return StrategyDirective.from_payload(
                {
                    "macro_directive": "DEFENSIVE_TURTLE",
                    "priority_targets": ["Bomber", "Artillery", "Tank"],
                    "resource_weights": {
                        "defense": 0.9,
                        "ground": max(0.5, strategy.resource_weights.get("ground", 0.4)),
                        "air": strategy.resource_weights.get("air", 0.3),
                    },
                    "aggression": min(strategy.aggression, 0.35),
                    "risk_tolerance": min(strategy.risk_tolerance, 0.25),
                }
            )
        if threats.get("enemy_bomber_count"):
            return StrategyDirective.from_payload(
                {
                    "macro_directive": "AIR_SUPERIORITY_RUSH",
                    "priority_targets": ["Bomber", "Airbase"],
                    "resource_weights": {"air": 0.8, "ground": 0.25, "defense": 0.65},
                    "aggression": max(strategy.aggression, 0.55),
                    "risk_tolerance": min(strategy.risk_tolerance, 0.35),
                }
            )
        if low_enemy_base <= 160 and (
            force_mix.get("friendly_air", 0) or force_mix.get("friendly_ground", 0) >= 8
        ):
            return StrategyDirective.from_payload(
                {
                    "macro_directive": "BASE_ASSAULT_AIR",
                    "priority_targets": ["Base", "Airbase", "Factory"],
                    "resource_weights": {"air": 0.65, "ground": 0.45, "economy": 0.05},
                    "aggression": 0.95,
                    "risk_tolerance": 0.65,
                }
            )
        return strategy
