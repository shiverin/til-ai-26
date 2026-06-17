"""WorldMemory: persistent cross-turn state + per-turn observation ingestion (PLAN B2).

The FastAPI process lives for the whole game, so one WorldMemory instance is our
memory. Terrain is static and accumulates forever; enemy buildings are remembered
until observed destroyed; enemy units decay after a few turns unseen.

Chat policy (PLAN B3 step 1): we read ONLY __system__ messages — private DMs
(treaty notices) and global broadcasts (eliminations). Player-authored chat is
never iterated beyond a cheap sender check and never stored.
"""

from __future__ import annotations

from engine.actions import ProduceUnitAction
from engine.constants import BUILDING_STATS
from engine.hex_grid import HexCoord, HexGrid

ENEMY_UNIT_MEMORY_TURNS = 10
_ELIM_SUFFIX = " has been eliminated"


def coord_of(e: dict) -> HexCoord:
    return HexCoord(e["q"], e["r"])


class WorldMemory:
    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        # identity / reset guard
        self.player_id: str | None = None
        self.map_w = 0
        self.map_h = 0
        self.grid: HexGrid | None = None
        self.turn = -1
        self.max_turns = 300

        # persistent knowledge
        self.terrain: dict[HexCoord, str] = {}
        self.rich_tiles: set[HexCoord] = set()
        self.enemy_buildings: dict[str, dict] = {}  # id -> entity dict + last_seen
        self.enemy_units: dict[str, dict] = {}  # id -> entity dict + last_seen
        self.eliminated: set[str] = set()
        self.elim_names: set[str] = set()  # unmapped elimination names
        self.seen_player_ids: set[str] = set()
        self.betrayers: set[str] = set()  # partners who initiated a break on us
        self.proposals_sent: dict[str, int] = {}  # pid -> turn last proposed
        self.partner_perimeter: dict[str, int] = {}  # pid -> camping score
        self.production_ledger: list[dict] = []  # building_id, unit_type, due_turn, target
        self.base_graveyard: list[tuple[HexCoord, int]] = []  # (coord, turn lost)
        self._prev_base_coords: set[HexCoord] = set()
        self.scout_targets: dict[str, HexCoord] = {}
        self.expansion_goal: HexCoord | None = None
        self.last_base_order_turn = -999
        self.home: HexCoord | None = None
        self._global_chat_idx = 0
        self._private_chat_idx = 0
        self._cost_map: dict[HexCoord, int] = {}

        # per-turn (rebuilt every ingest)
        self.gold = 0
        self.visible: set[HexCoord] = set()
        self.occupied: dict[HexCoord, dict] = {}
        self.own_units: list[dict] = []
        self.own_buildings: list[dict] = []
        self.bases: list[dict] = []  # completed own Bases
        self.visible_enemies: list[dict] = []
        self.treaties: dict[str, dict] = {}  # partner_id -> treaty dict
        self.incoming_proposals: list[dict] = []
        self.known_players: list[str] = []

    # ── ingestion ──────────────────────────────────────────────────────────────

    def ingest(self, obs: dict) -> None:
        pid = obs.get("player_id")
        w = obs.get("map_width", 35)
        h = obs.get("map_height", 30)
        turn = obs.get("turn_number", 0)
        if pid != self.player_id or w != self.map_w or h != self.map_h or turn < self.turn:
            self._reset()
            self.player_id = pid
            self.map_w, self.map_h = w, h
            self.grid = HexGrid(w, h)
            # pad every tile's move cost with 2 until its terrain is known
            self._cost_map = {c: 2 for c in self.grid.all_coords()}
        self.turn = turn
        self.max_turns = obs.get("max_turns", 300)
        self.gold = obs.get("resources", {}).get("gold", 0)
        self.known_players = list(obs.get("known_players", []))

        self.visible = set()
        self.occupied = {}
        self.own_units = []
        self.own_buildings = []
        self.bases = []
        self.visible_enemies = []

        for tile in obs.get("visible_tiles", []):
            c = HexCoord(tile["q"], tile["r"])
            self.visible.add(c)
            terr = tile.get("terrain", "normal")
            self.terrain[c] = terr
            self._cost_map[c] = 2 if terr == "difficult" else 1
            if terr == "rich_resource":
                self.rich_tiles.add(c)
            for e in tile.get("entities", []):
                self.occupied[coord_of(e)] = e
                owner = e.get("owner_id")
                if owner == self.player_id:
                    if e.get("type") in BUILDING_STATS:
                        self.own_buildings.append(e)
                        if e["type"] == "Base" and e.get("is_complete"):
                            self.bases.append(e)
                    else:
                        self.own_units.append(e)
                else:
                    self.seen_player_ids.add(owner)
                    self.visible_enemies.append(e)
                    rec = dict(e)
                    rec["last_seen"] = turn
                    if e.get("type") in BUILDING_STATS:
                        self.enemy_buildings[e["id"]] = rec
                    else:
                        self.enemy_units[e["id"]] = rec

        if self.home is None and self.bases:
            self.home = coord_of(self.bases[0])

        # graveyard: a Base coord we held last turn with no Base on it now means
        # the Base died (own entities are always visible) — remember where, so
        # rebuilds stop walking into the same hunter's kill zone (cb31973a:
        # rebuilt 3 tiles from a fresh base kill and lost the rebuild in 11 turns)
        cur_base_coords = {
            coord_of(b) for b in self.own_buildings if b["type"] == "Base"
        }
        for c in self._prev_base_coords - cur_base_coords:
            self.base_graveyard.append((c, turn))
        self.base_graveyard = self.base_graveyard[-32:]
        self._prev_base_coords = cur_base_coords

        # forget remembered enemies disproven by current vision
        for reg in (self.enemy_buildings, self.enemy_units):
            stale = [
                eid
                for eid, rec in reg.items()
                if coord_of(rec) in self.visible
                and self.occupied.get(coord_of(rec), {}).get("id") != eid
            ]
            for eid in stale:
                del reg[eid]
        # decay unit memory
        old = [
            eid
            for eid, rec in self.enemy_units.items()
            if turn - rec["last_seen"] > ENEMY_UNIT_MEMORY_TURNS
        ]
        for eid in old:
            del self.enemy_units[eid]

        self.treaties = {t["partner_id"]: t for t in obs.get("treaties", [])}
        self.incoming_proposals = list(obs.get("incoming_treaty_proposals", []))

        self._read_system_chat(obs)
        # production ledger: drop entries past due (spawned or lost)
        self.production_ledger = [
            p for p in self.production_ledger if p["due_turn"] >= turn
        ]

    def _read_system_chat(self, obs: dict) -> None:
        """Scan only NEW messages, and only __system__ ones (B3 step 1)."""
        gchat = obs.get("global_chat", [])
        for m in gchat[self._global_chat_idx :]:
            if m.get("sender_id") != "__system__":
                continue
            text = m.get("text", "")
            if text.endswith(_ELIM_SUFFIX):
                name = text[: -len(_ELIM_SUFFIX)]
                # broadcast carries the display NAME; ids match names in the
                # local harness — corroborate against every id we know of
                if name in self.seen_player_ids or name in self.known_players:
                    self.eliminated.add(name)
                else:
                    self.elim_names.add(name)
        self._global_chat_idx = len(gchat)

        pchat = obs.get("private_chat", [])
        for m in pchat[self._private_chat_idx :]:
            if m.get("sender_id") != "__system__":
                continue
            text = m.get("text", "")
            if "is breaking the peace treaty" in text or "initiated a treaty break" in text:
                # "X is breaking the peace treaty with Y — 5 turns until war"
                name = text.split(" is breaking", 1)[0].split(" initiated", 1)[0]
                if name in self.seen_player_ids or name in self.known_players:
                    self.betrayers.add(name)
        self._private_chat_idx = len(pchat)

    # ── queries ────────────────────────────────────────────────────────────────

    def at_peace_with(self, owner_id: str | None) -> bool:
        """True while a treaty (ACTIVE or BREAKING) makes attacks invalid no-ops."""
        return owner_id in self.treaties

    def hostile_units(self, include_partners: bool = False) -> list[dict]:
        """Remembered + visible enemy units that threaten us: anyone not under
        treaty cover (or everyone, during war-prep). Dead players' units decay."""
        return [
            rec
            for rec in self.enemy_units.values()
            if (include_partners or not self.at_peace_with(rec.get("owner_id")))
            and rec.get("owner_id") not in self.eliminated
        ]

    def move_costs(self) -> dict[HexCoord, int]:
        return self._cost_map

    def pending_spawns_for(self, building_id: str) -> int:
        return sum(1 for p in self.production_ledger if p["building_id"] == building_id)

    def reconcile_ledger(self, turn: int, final_actions: list) -> None:
        """Drop this-turn ledger entries whose produce order didn't survive
        validation — phantom entries otherwise block the spawn-tile picker for
        build_turns turns (and on a 2-tile ring that is a permanent stall)."""
        survived = [
            (a.building_id, a.unit_type, a.target)
            for a in final_actions
            if isinstance(a, ProduceUnitAction)
        ]
        kept = []
        for p in self.production_ledger:
            if p.get("ordered_turn") != turn:
                kept.append(p)
                continue
            key = (p["building_id"], p["unit_type"], p["target"])
            if key in survived:
                survived.remove(key)
                kept.append(p)
        self.production_ledger = kept

    def our_unit_count(self, unit_type: str) -> int:
        n = sum(1 for u in self.own_units if u["type"] == unit_type)
        n += sum(1 for p in self.production_ledger if p["unit_type"] == unit_type)
        return n

    def building_count(self, building_type: str, complete_only: bool = False) -> int:
        return sum(
            1
            for b in self.own_buildings
            if b["type"] == building_type and (b.get("is_complete") or not complete_only)
        )
