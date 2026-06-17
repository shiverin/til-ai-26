"""Scout tasking (PLAN C-Scouting): greedy frontier exploration until ~70% of
the map is known, then convert to invisible watchtowers — park in concealment
on the corridor between home and the nearest known enemy base."""

from __future__ import annotations

from engine.hex_grid import HexCoord

from world import WorldMemory, coord_of

WATCHTOWER_KNOWN_FRACTION = 0.7


def assign(world: WorldMemory, war_prep: bool) -> dict[str, HexCoord]:
    """unit_id -> destination for every Scout we own."""
    grid = world.grid
    scouts = [u for u in world.own_units if u["type"] == "Scout"]
    if grid is None or not scouts:
        return {}

    out: dict[str, HexCoord] = {}
    if world.expansion_goal is not None:
        # one scout escorts the expansion: a Base build needs build-time vision
        out[scouts[0]["id"]] = world.expansion_goal
        scouts = scouts[1:]
        if not scouts:
            return out

    known_fraction = len(world.terrain) / max(1, world.map_w * world.map_h)
    if war_prep or known_fraction >= WATCHTOWER_KNOWN_FRACTION:
        out.update(_watchtowers(world, scouts))
    else:
        out.update(_frontier(world, scouts))
    return out


def _frontier(world: WorldMemory, scouts: list[dict]) -> dict[str, HexCoord]:
    grid = world.grid
    unknown = [c for c in grid.all_coords() if c not in world.terrain]
    if not unknown:
        return _watchtowers(world, scouts)

    out: dict[str, HexCoord] = {}
    taken: list[HexCoord] = []
    for s in scouts:
        sc = coord_of(s)
        prev = world.scout_targets.get(s["id"])
        if prev is not None and prev not in world.terrain:
            out[s["id"]] = prev  # keep pushing toward a still-unknown goal
            taken.append(prev)
            continue
        # nearest unknown tile, repelled from other scouts' goals so the two
        # scouts sweep opposite directions
        best, best_score = None, 10**9
        for c in unknown:
            score = grid.distance(sc, c)
            for t in taken:
                score -= 0.4 * grid.distance(c, t)
            for other in scouts:
                if other["id"] != s["id"]:
                    score -= 0.2 * grid.distance(c, coord_of(other))
            if score < best_score:
                best, best_score = c, score
        if best is not None:
            out[s["id"]] = best
            taken.append(best)
            world.scout_targets[s["id"]] = best
    return out


def _watchtowers(world: WorldMemory, scouts: list[dict]) -> dict[str, HexCoord]:
    """Concealment tiles between our bases and the nearest known enemy base."""
    grid = world.grid
    home = world.home or (coord_of(world.bases[0]) if world.bases else None)
    if home is None:
        return {}
    enemy_bases = [
        rec
        for rec in world.enemy_buildings.values()
        if rec["type"] == "Base" and rec.get("owner_id") not in world.eliminated
    ]
    if enemy_bases:
        nearest = min(enemy_bases, key=lambda r: grid.distance(home, coord_of(r)))
        mid_line = grid.line(home, coord_of(nearest))
        anchor = mid_line[len(mid_line) // 2]
    else:
        anchor = home
    conceal = [c for c, t in world.terrain.items() if t == "concealment"]
    out: dict[str, HexCoord] = {}
    taken: set[HexCoord] = set()
    for s in scouts:
        cands = [c for c in conceal if c not in taken and c not in world.occupied]
        if not cands:
            out[s["id"]] = anchor
            continue
        best = min(
            cands,
            key=lambda c: grid.distance(anchor, c) + 0.3 * grid.distance(home, c),
        )
        out[s["id"]] = best
        taken.add(best)
        world.scout_targets[s["id"]] = best
    return out
