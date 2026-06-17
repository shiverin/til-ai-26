"""Movement planning (PLAN B3 step 5): kiting retreats → garrison wall slots →
rallies → scout exploration.

Engine facts this code leans on (verified in turn_processor):
- Attacks fire pre-move from the CURRENT tile, so a unit may attack and move in
  the same turn (shoot-and-scoot).
- Intermediate path tiles are never occupancy-checked — only the destination.
  Pathfinding therefore ignores traversal blocking and only needs a free endpoint.
- Destination conflicts are checked against PRE-move positions, so we never step
  onto a tile a friendly is vacating this turn.
"""

from __future__ import annotations

from engine.actions import MoveAction
from engine.constants import TREATY_CUTOFF_TURN
from engine.hex_grid import HexCoord

from threats import L1, ThreatReport
from world import WorldMemory, coord_of

INTERCEPT_RADIUS = 3  # hostiles this close to a base are besiegers — kill them


def plan_moves(
    world: WorldMemory,
    threat: ThreatReport,
    engaged: dict[str, object],
    reserved: set[HexCoord],
    scout_goals: dict[str, HexCoord],
) -> list[MoveAction]:
    grid = world.grid
    if grid is None:
        return []
    moves: list[MoveAction] = []
    moved: set[str] = set()
    blocked = set(world.occupied.keys())  # endpoints only; traversal is free
    hostiles = [
        h for h in threat.hostiles if h.get("last_seen") == world.turn
    ] or threat.hostiles
    # spawn-or-lose-gold: idling units must not squat on production spawn rings;
    # nor on the build band (tiles adjacent to own buildings) — saturating it
    # starves construction entirely (seen in self-play)
    spawn_ring: set[HexCoord] = set()
    build_band: set[HexCoord] = set()
    for b in world.own_buildings:
        build_band.update(grid.neighbors(coord_of(b)))
        if b["type"] in ("Barracks", "Factory", "Airbase"):
            spawn_ring.update(grid.neighbors(coord_of(b)))
    idle_avoid = spawn_ring | build_band
    # pending spawn tiles must be clear on the due turn (spawn-or-lose-gold):
    # reserving them makes step 0 evict squatters and stops anyone ending here
    for p in world.production_ledger:
        reserved.add(p["target"])
    # rally/picket radius grows with the army: packing 100+ units into a
    # radius-4 disk (~60 tiles) is what clogged spawn rings and silently
    # killed 8 Fighters at completion in c2f5fc26
    n_own = len(world.own_units)
    spread = max(4, int((n_own / 2) ** 0.5) + 2)
    picket_r = max(3, spread - 1)

    def commit(unit: dict, path: list[HexCoord]) -> None:
        moves.append(MoveAction(unit_id=unit["id"], path=path))
        moved.add(unit["id"])
        # NOTE: the origin stays blocked — the engine checks destination
        # conflicts against PRE-move positions, so stepping onto a tile a
        # friendly is vacating this turn is always a silent no-op
        blocked.add(path[-1])
        reserved.add(path[-1])

    # ── 0. vacate-and-build: evict own units from tiles economy claimed ───────
    for u in world.own_units:
        if u["id"] in moved or coord_of(u) not in reserved:
            continue
        uc = coord_of(u)
        for c in sorted(
            grid.neighbors(uc),
            key=lambda c: (c in spawn_ring, world.move_costs().get(c, 2)),
        ):
            if (
                c not in blocked
                and c not in reserved
                and world.move_costs().get(c, 2) <= u.get("movement_range", 0)
            ):
                commit(u, [uc, c])
                break

    # ── 0.5 intercept besiegers: a lone Bomber adjacent to a Base kills it in
    # two turns, and wall units two tiles away can't touch it (f276a8bb: base
    # (4,21) died with 24 passive guards). Fighters hunt anything sieging any
    # base; assignment is capped so a feint can't peel the whole defense.
    # a Base under construction is a 5-turn 300g sitting duck — it gets the
    # same intercept cover as a finished one (cb31973a: the t204 rebuild died
    # at t216 to a pack nobody contested)
    protect = world.bases + [
        b
        for b in world.own_buildings
        if b["type"] == "Base" and not b.get("is_complete")
    ]
    besiegers = [
        h
        for h in hostiles
        if any(
            grid.distance(coord_of(h), coord_of(b)) <= INTERCEPT_RADIUS
            for b in protect
        )
    ]
    if besiegers:
        # 3 Fighters per Bomber is exactly one-shot lethal (3×50 ≥ 150 HP) —
        # bombers are the base-killers and must die the turn they arrive
        quota = {h["id"]: (3 if h["type"] == "Bomber" else 2) for h in besiegers}
        for u in world.own_units:
            if u["type"] != "Fighter" or u["id"] in moved:
                continue
            uc = coord_of(u)
            open_prey = [h for h in besiegers if quota[h["id"]] > 0]
            if not open_prey:
                break
            prey = min(
                open_prey, key=lambda h: grid.distance(uc, coord_of(h))
            )
            quota[prey["id"]] -= 1
            pc = coord_of(prey)
            if grid.distance(uc, pc) <= u.get("attack_range", 2):
                moved.add(u["id"])  # in range — hold; combat fires pre-move
            else:
                path = _path_toward(
                    world, u, pc, u.get("attack_range", 2), blocked, reserved
                )
                if path:
                    commit(u, path)

    # ── 1. kiting / retreats for ranged units ─────────────────────────────────
    # Artillery near a base is the wall gun: it stands and fires (kiting it away
    # is how we lost bases to fighter swarms). Only field artillery retreats.
    base_coords = [coord_of(b) for b in world.bases]
    for u in world.own_units:
        if u["id"] in moved or u.get("attack_range", 0) < 2:
            continue
        if (
            u["type"] == "Artillery"
            and coord_of(u) not in spawn_ring  # never squat a spawn tile
            and any(grid.distance(coord_of(u), bc) <= 3 for bc in base_coords)
        ):
            moved.add(u["id"])  # hold the gun line
            continue
        path = _kite_path(world, u, hostiles, engaged.get(u["id"]), blocked, reserved)
        if path:
            commit(u, path)

    # ── 2. garrison: infantry/medics man the walls of their nearest base ──────
    # At war, concentrate: 8 bases × 1/8 of the army died piecemeal to massed
    # air (1ff4e9f5). Defend the 3 bases nearest the production cluster with
    # everything; outlying bases are expendable extra lives, not hills to die on.
    at_war = threat.war_prep or any(
        lv >= 2 for lv in threat.base_levels.values()
    )
    defend_bases = world.bases
    if at_war and len(world.bases) > 3:
        prod_coords = [
            coord_of(b)
            for b in world.own_buildings
            if b["type"] in ("Barracks", "Factory", "Airbase")
        ]

        def keep_score(b: dict) -> int:
            bc = coord_of(b)
            return sum(sorted(grid.distance(bc, p) for p in prod_coords)[:4])

        defend_bases = sorted(world.bases, key=keep_score)[:3]
    foot = [
        u
        for u in world.own_units
        if u["type"] in ("Infantry", "Medic", "Tank", "Artillery")
        and u["id"] not in moved
    ]
    for base in sorted(
        defend_bases, key=lambda b: -threat.base_levels.get(b["id"], 0)
    ):
        bc = coord_of(base)
        mine = [u for u in foot if u["id"] not in moved]
        mine.sort(key=lambda u: grid.distance(coord_of(u), bc))
        level = threat.base_levels.get(base["id"], 0)
        keep = 12 if level >= 2 else (6 if level >= L1 else 3)
        local_prey = [
            h
            for h in besiegers
            if grid.distance(coord_of(h), bc) <= INTERCEPT_RADIUS + 1
        ]
        melee_quota = {h["id"]: 3 for h in local_prey}
        for u in mine[:keep]:
            uc = coord_of(u)
            # melee intercept: close the 1-tile gap to a besieger instead of
            # standing on a wall slot it can't be hit from
            if u["type"] in ("Infantry", "Tank") and local_prey:
                open_prey = [h for h in local_prey if melee_quota[h["id"]] > 0]
                if open_prey:
                    prey = min(
                        open_prey,
                        key=lambda h: grid.distance(uc, coord_of(h)),
                    )
                    pc = coord_of(prey)
                    if grid.distance(uc, pc) <= 1:
                        melee_quota[prey["id"]] -= 1
                        moved.add(u["id"])  # adjacent — hold and fight
                        continue
                    path = _path_toward(world, u, pc, 1, blocked, reserved)
                    if path:
                        melee_quota[prey["id"]] -= 1
                        commit(u, path)
                        continue
            if level >= L1:
                # man a wall slot (free, non-difficult-for-infantry base neighbour)
                slots = [
                    c
                    for c in grid.neighbors(bc)
                    if (c not in blocked or c == uc)
                    and c not in reserved
                    and c not in spawn_ring  # walls must not choke production
                    and not (
                        u["type"] != "Tank"
                        and world.terrain.get(c) == "difficult"
                        and u.get("movement_range", 1) < 2
                    )
                ]
                if uc in slots or uc == bc:
                    moved.add(u["id"])  # already in position — hold
                    continue
                if slots:
                    # prefer slots that don't choke a production spawn ring
                    dest = min(
                        slots,
                        key=lambda c: (c in spawn_ring, grid.distance(uc, c)),
                    )
                    path = _path(world, u, dest, blocked, reserved)
                    if path:
                        commit(u, path)
                        continue
            # loiter near the base as a picket, clear of the build/spawn band
            if grid.distance(uc, bc) > picket_r or uc in idle_avoid:
                dest = _idle_dest(world, uc, bc, picket_r, blocked, reserved, idle_avoid)
                if dest is None:
                    dest = _idle_dest(world, uc, bc, picket_r, blocked, reserved, spawn_ring)
                path = (
                    _path(world, u, dest, blocked, reserved)
                    if dest
                    else _path_toward(world, u, bc, 2, blocked, reserved)
                )
                if path:
                    commit(u, path)
            else:
                moved.add(u["id"])

    # ── 3. scouts ─────────────────────────────────────────────────────────────
    for u in world.own_units:
        if u["type"] != "Scout" or u["id"] in moved:
            continue
        goal = scout_goals.get(u["id"])
        if goal is None or coord_of(u) == goal:
            continue
        path = _path_toward(world, u, goal, 0, blocked, reserved)
        if path:
            commit(u, path)

    # ── 4. bombers: rally at home until the void, then kill enemy AIR PRODUCTION
    # (an Airbase printing 14 fresh Bombers outlasts any base-trading — gpt in
    # 1ff4e9f5); excess fighters fly escort instead of idling at home
    target = _bomber_objective(world)
    strike_on = target is not None and world.turn >= min(
        TREATY_CUTOFF_TURN, world.max_turns - 50
    )
    for u in world.own_units:
        if u["type"] != "Bomber" or u["id"] in moved:
            continue
        if strike_on:
            path = _path_toward(world, u, target, 1, blocked, reserved)
            if path:
                commit(u, path)
        elif world.home is not None and grid.distance(coord_of(u), world.home) > 2:
            path = _path_toward(world, u, world.home, 2, blocked, reserved)
            if path:
                commit(u, path)
    if strike_on:
        idle_fighters = [
            u
            for u in world.own_units
            if u["type"] == "Fighter" and u["id"] not in moved
        ]
        enemy_air = sum(
            1
            for r in world.enemy_units.values()
            if r["type"] in ("Fighter", "Bomber")
            and r.get("owner_id") not in world.eliminated
        )
        home_floor = max(6, enemy_air // 2)  # air defense stays staffed
        for u in idle_fighters[home_floor:]:
            path = _path_toward(world, u, target, 2, blocked, reserved)
            if path:
                commit(u, path)

    # ── 5. everyone else rallies toward the most relevant base ────────────────
    if at_war and defend_bases:
        # rally to the most pressed KEEP base — never to an expendable outpost
        def press(b: dict) -> int:
            bc = coord_of(b)
            return min(
                (grid.distance(bc, coord_of(h)) for h in hostiles), default=99
            )

        anchor = coord_of(min(defend_bases, key=press))
    else:
        anchor = (
            coord_of(threat.threatened_base) if threat.threatened_base else world.home
        )
    if anchor is not None:
        for u in world.own_units:
            if u["id"] in moved or u.get("movement_range", 0) < 1:
                continue
            uc = coord_of(u)
            if grid.distance(uc, anchor) > spread or uc in idle_avoid:
                dest = _idle_dest(world, uc, anchor, spread, blocked, reserved, idle_avoid)
                if dest is None:
                    dest = _idle_dest(world, uc, anchor, spread, blocked, reserved, spawn_ring)
                path = (
                    _path(world, u, dest, blocked, reserved)
                    if dest
                    else _path_toward(world, u, anchor, 2, blocked, reserved)
                )
                if path:
                    commit(u, path)
    return moves


def _idle_dest(world, uc, anchor, radius, blocked, reserved, spawn_ring):
    """Nearest free tile within `radius` of anchor that keeps spawn rings clear."""
    grid = world.grid
    best, best_d = None, 10**9
    for c in grid.disk(anchor, radius):
        if c == uc or c in blocked or c in reserved or c in spawn_ring:
            continue
        d = grid.distance(uc, c)
        if d < best_d:
            best, best_d = c, d
    return best


# ── helpers ────────────────────────────────────────────────────────────────────


def _reach(unit: dict) -> int:
    return unit.get("movement_range", 0) + unit.get("attack_range", 0)


def _margin(world: WorldMemory, c: HexCoord, hostiles: list[dict]) -> int:
    """How far outside every hostile's next-turn strike envelope c is."""
    return min(
        (world.grid.distance(c, coord_of(h)) - _reach(h) for h in hostiles),
        default=99,
    )


def _candidate_tiles(world: WorldMemory, unit: dict, blocked, reserved):
    grid = world.grid
    origin = coord_of(unit)
    cands = grid.reachable(origin, unit.get("movement_range", 0), world.move_costs())
    cands = [c for c in cands if c not in blocked and c not in reserved]
    cands.append(origin)
    return cands


def _kite_path(world, unit, hostiles, fired_at, blocked, reserved):
    """Shoot-and-scoot: end the turn outside enemy reach, preferring tiles that
    keep the fired-at target inside our own next-turn envelope."""
    if not hostiles:
        return None
    grid = world.grid
    origin = coord_of(unit)
    if _margin(world, origin, hostiles) >= 1 and fired_at is None:
        return None  # already safe and idle — rally logic can have it
    own_reach = _reach(unit)
    best, best_key = None, None
    for c in _candidate_tiles(world, unit, blocked, reserved):
        m = min(_margin(world, c, hostiles), 2)  # safe-enough is safe
        re_engage = 0
        if fired_at is not None:
            re_engage = 1 if grid.distance(c, fired_at) <= own_reach else 0
        home_d = grid.distance(c, world.home) if world.home else 0
        key = (m, re_engage, -home_d)
        if best_key is None or key > best_key:
            best, best_key = c, key
    if best is None or best == origin:
        return None
    return _path(world, unit, best, blocked, reserved)


def _path(world, unit, dest, blocked, reserved):
    """A* path truncated to this unit's movement budget; endpoint must be free."""
    grid = world.grid
    origin = coord_of(unit)
    if origin == dest:
        return None
    full = grid.shortest_path(origin, dest, world.move_costs())
    if not full or len(full) < 2:
        return None
    budget = unit.get("movement_range", 0)
    path = [origin]
    spent = 0
    for step in full[1:]:
        cost = world.move_costs().get(step, 2)
        if spent + cost > budget or len(path) > budget:
            break
        path.append(step)
        spent += cost
    # back off any endpoint that is occupied or reserved
    while len(path) > 1 and (path[-1] in blocked or path[-1] in reserved):
        path.pop()
    return path if len(path) > 1 else None


def _path_toward(world, unit, goal, stop_within, blocked, reserved):
    """Move toward goal, content to stop within `stop_within` of it."""
    grid = world.grid
    if grid.distance(coord_of(unit), goal) <= stop_within:
        return None
    return _path(world, unit, goal, blocked, reserved)


_STRIKE_PRIORITY = {"Airbase": 0, "Factory": 1, "Base": 2}


def _bomber_objective(world: WorldMemory):
    """Air production first (Airbase > Factory), then Bases: killing the
    building that prints Bombers stops the bleeding; a Base is just one of
    the victim's extra lives."""
    grid = world.grid
    home = world.home
    cands = [
        rec
        for rec in world.enemy_buildings.values()
        if rec["type"] in _STRIKE_PRIORITY
        and rec.get("owner_id") not in world.eliminated
        and not world.at_peace_with(rec.get("owner_id"))
    ]
    if not cands or home is None:
        return None
    best = min(
        cands,
        key=lambda r: (
            _STRIKE_PRIORITY[r["type"]],
            grid.distance(home, coord_of(r)),
        ),
    )
    return coord_of(best)
