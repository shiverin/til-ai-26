"""The menu of cascade layers. Each public layer is a pure function
(belief, danger, planner, params) -> action_or_None. Composed into named
strategies by strategies.py and run by decide.act().
"""
import math
from collections import deque

from scripted.blast import bomb_reaches
from scripted.geometry import (
    BACKWARD, FORWARD, LEFT, MOVE, PLACE_BOMB, RIGHT, STAY, chebyshev,
)
from scripted.pathfind import BOMB_TIMER, build_planner

INF = float("inf")
ESCAPE_HORIZON = BOMB_TIMER          # steps to vacate a blast zone (= bomb fuse)
BASE_MAX_HEALTH = 100.0              # enemy base HP at full (til_environment default)
BOMB_ATTACK = 20.0                   # damage one bomb deals
OPENNESS_WINDOW = 12     # closest safe cells scored for openness (bounds per-tick cost)


def _openness(belief, danger, cell, radius):
    """Count safe cells reachable from `cell` within `radius` BFS steps (cell
    itself included). A dead-end pocket scores low; an open area scores high.
    Walls block traversal; danger cells are neither counted nor traversed."""
    gs = belief.prior.grid_size
    start = tuple(cell)
    seen = {start}
    q = deque([(start, 0)])
    while q:
        t, d = q.popleft()
        if d >= radius:
            continue
        for mdir in range(4):
            dx, dy = MOVE[mdir]
            nb = (t[0] + dx, t[1] + dy)
            if (nb in seen or not (0 <= nb[0] < gs and 0 <= nb[1] < gs)
                    or belief.is_wall(t, nb) or danger.is_dangerous(nb)):
                continue
            seen.add(nb)
            q.append((nb, d + 1))
    return len(seen)


def survive(belief, danger, planner, params):
    """Layer 1 — escape bomb danger.

    Tier 1: when a fully-safe cell is reachable before our cell detonates,
    route there, biasing toward open cells over dead-end pockets; first drop a
    surplus bomb if the escape still completes afterwards. Tier 2: when no
    fully-safe cell is reachable in time, move to the least-bad reachable cell
    (fewest overlapping bombs, then latest detonation, then nearest), or yield
    to the objective layers when already on it or when nothing is reachable.
    Never returns STAY.
    """
    loc = belief.location
    if not danger.is_dangerous(loc, within=ESCAPE_HORIZON):
        return None
    gs = belief.prior.grid_size
    deadline = danger.ticks_to_danger(loc)

    # Tier 1 — fully-safe cells reachable before loc detonates.
    safe = []
    for x in range(gs):
        for y in range(gs):
            cell = (x, y)
            if danger.is_dangerous(cell):
                continue
            # Strict: the agent must arrive strictly before the detonation tick.
            if planner.steps_to(cell) < deadline:
                safe.append(cell)
    if safe:
        # Closest first; score openness only for the closest few (bounds cost).
        safe.sort(key=planner.dist_to)
        chosen, best_score = None, INF
        for cell in safe[:OPENNESS_WINDOW]:
            score = (planner.dist_to(cell)
                     - params.openness_weight
                     * _openness(belief, danger, cell, params.openness_radius))
            if score < best_score:
                best_score, chosen = score, cell
        # Opportunistic bomb drop: a surplus bomb, the escape still completes
        # after the place tick — and, while live bases remain, the dropped
        # bomb must actually reach a base or a live enemy (no zero-value drop).
        live_bases = belief.live_enemy_bases()
        drop_hits = (not live_bases
                     or any(bomb_reaches(loc, bs, belief) for bs in live_bases)
                     or any(bomb_reaches(loc, e, belief)
                            for e in belief.live_enemies()))
        if (belief.team_bombs >= params.bomb_drop_min
                and planner.steps_to(chosen) + 1 + params.bomb_drop_buffer
                <= deadline
                and drop_hits):
            return PLACE_BOMB
        # `chosen` is a non-dangerous cell while `loc` is dangerous, so
        # `chosen != loc` and `first_action` is always a real move here.
        return planner.first_action(chosen)

    # Tier 2 — no fully-safe escape in time; move to the least-bad cell.
    best_key, best_tile = None, None
    for x in range(gs):
        for y in range(gs):
            cell = (x, y)
            d = planner.dist_to(cell)
            if d == INF:
                continue
            key = (danger.overlap(cell), -danger.ticks_to_danger(cell), d)
            if best_key is None or key < best_key:
                best_key, best_tile = key, cell
    if best_tile is None or best_tile == loc:
        return None
    return planner.first_action(best_tile)


def _effective_hp(belief, base):
    """Believed HP of `base` after the agent's own in-flight bombs land.

    Observed HP (last-seen ratio x BASE_MAX_HEALTH) minus BOMB_ATTACK per own
    bomb whose blast reaches the base. Floored at 0. This is the true
    remaining work — observed HP does not drop until a bomb's fuse expires.
    A base never observed is assumed full (conservative — never skipped).
    """
    observed = belief.enemy_base_health.get(base, 1.0) * BASE_MAX_HEALTH
    in_flight = sum(1 for cell, _ in belief.own_bombs
                    if bomb_reaches(cell, base, belief))
    return max(0.0, observed - BOMB_ATTACK * in_flight)


def _base_doomed(belief, base):
    """True if the agent's own in-flight bombs already finish `base`, so
    another bomb on it would be wasted."""
    return _effective_hp(belief, base) <= 0.0


def _target_base(belief, planner, params):
    """Pick the single enemy base the agent commits to.

    Returns (base, effective_hp, bombs_needed) for the live, non-doomed base
    with the lowest blended score `bombs_needed + target_travel_weight *
    arrival` (arrival = earliest planner tick to a tile that can bomb it).
    Ties, and the all-unreachable case, fall to the lowest bombs_needed. A
    damaged base keeps the lowest bombs_needed, so the target is sticky.
    Returns None when no live, non-doomed base exists.
    """
    gs = belief.prior.grid_size
    best_key, best = None, None
    for base in belief.live_enemy_bases():
        if _base_doomed(belief, base):
            continue
        eff = _effective_hp(belief, base)
        bombs_needed = math.ceil(eff / BOMB_ATTACK)
        arrival = INF
        for x in range(gs):
            for y in range(gs):
                if bomb_reaches((x, y), base, belief):
                    d = planner.dist_to((x, y))
                    if d < arrival:
                        arrival = d
        score = bombs_needed + params.target_travel_weight * arrival
        key = (score, bombs_needed, arrival)
        if best_key is None or key < best_key:
            best_key, best = key, (base, eff, bombs_needed)
    return best


def strike(belief, danger, planner, params):
    """Bomb the chosen enemy base, in two phases by its effective HP.

    Soften (effective_hp > soften_floor): a near-full base is unstealable, so
    commit with any bomb in hand and drive its HP down. Leashed one-shot
    (effective_hp <= soften_floor): a crippled base is stealable, so commit
    only with the full kill in hand (team_bombs >= bombs_needed) — otherwise
    yield, letting sweep accumulate bombs near the base.

    Within a committed phase: bomb a base in direct range, else breach a wall
    toward it when that is strictly faster, else navigate to a hit-tile. A
    bomb hits a base only within Chebyshev 2 AND with line-of-sight.
    """
    if belief.team_bombs <= 0:
        return None
    target = _target_base(belief, planner, params)
    if target is None:
        return None
    base, eff_hp, bombs_needed = target

    # Phase-B gate: a stealable base is struck only with the full kill in hand.
    if eff_hp <= params.soften_floor and belief.team_bombs < bombs_needed:
        return None

    loc = belief.location
    # 1. Direct hit from where we stand.
    if bomb_reaches(loc, base, belief):
        return PLACE_BOMB

    gs = belief.prior.grid_size

    def hit_dist(plan):
        """Earliest arrival tick at any tile that can bomb `base`."""
        best = INF
        for x in range(gs):
            for y in range(gs):
                if bomb_reaches((x, y), base, belief):
                    d = plan.dist_to((x, y))
                    if d < best:
                        best = d
        return best

    # 2. Breach: dropping a bomb now opens a wall and reaches a hit-tile
    #    strictly sooner.
    t_a = hit_dist(planner)
    if belief.team_bombs >= params.breach_min_bombs:
        bomb_planner = build_planner(belief, danger, place_bomb_first=True)
        if hit_dist(bomb_planner) < t_a:
            return PLACE_BOMB

    # 3. Navigate toward the nearest hit-tile (no breach).
    best, best_tile = INF, None
    for x in range(gs):
        for y in range(gs):
            if not bomb_reaches((x, y), base, belief):
                continue
            d = planner.dist_to((x, y))
            if d < best:
                best, best_tile = d, (x, y)
    if best_tile is None or best == INF:
        return None
    return planner.first_action(best_tile)


FORAGE_MOVES = (FORWARD, BACKWARD, LEFT, RIGHT)


def _move_result(belief, danger, tile, facing, action):
    """Apply one movement action. Returns (new_tile, new_facing), or None if
    the move is blocked, leaves the grid, or steps into a danger cell."""
    if action == LEFT:
        return (tile, (facing + 3) % 4)
    if action == RIGHT:
        return (tile, (facing + 1) % 4)
    mdir = facing if action == FORWARD else (facing + 2) % 4
    dx, dy = MOVE[mdir]
    nb = (tile[0] + dx, tile[1] + dy)
    gs = belief.prior.grid_size
    if not (0 <= nb[0] < gs and 0 <= nb[1] < gs):
        return None
    if (belief.is_wall(tile, nb) or danger.is_dangerous(nb)
            or nb in belief.frozen_enemies):
        return None
    return (nb, facing)


def forage(belief, danger, planner, params):
    """Endgame greedy collector — active once every enemy base is destroyed.

    Looks two moves ahead: for each first move, adds the collectible value of
    the tile it lands on to the best value reachable by one further move, then
    returns the first move of the best pair. Yields (returns None) when no
    collectible is within two moves, so the cascade falls through to sweep.
    """
    if params.forage_requires_endgame and belief.live_enemy_bases():
        return None                       # bases remain — not the endgame yet
    remaining = belief.remaining_collectibles()
    if params.camp_leash is not None:
        base = belief.prior.our_base
        remaining = {c: v for c, v in remaining.items()
                     if chebyshev(c, base) <= params.camp_leash}
    start_tile, facing = belief.location, belief.facing
    best_score, best_action = 0.0, None
    for a1 in FORAGE_MOVES:
        s1 = _move_result(belief, danger, start_tile, facing, a1)
        if s1 is None:
            continue
        v1 = remaining.get(s1[0], 0.0) if s1[0] != start_tile else 0.0
        best_after = 0.0
        for a2 in FORAGE_MOVES:
            s2 = _move_result(belief, danger, s1[0], s1[1], a2)
            if s2 is None:
                continue
            is_new = s2[0] not in (start_tile, s1[0])
            v2 = remaining.get(s2[0], 0.0) if is_new else 0.0
            best_after = max(best_after, v2)
        score = v1 + best_after
        if score > best_score:
            best_score, best_action = score, a1
    return best_action if best_score > 0.0 else None


def sweep(belief, danger, planner, params):
    """Head for the best-value reachable collectible.

    Leash precedence: the camper's our-base leash (`camp_leash`) wins; else a
    Phase-B target base (effective_hp <= soften_floor) leashes collection to
    `bombs_needed + 1` Chebyshev of that base, so the agent accumulates bombs
    near the kill; else no leash. The drift gradient points at the target base.

    score = value / (1 + dist) + a small gradient toward the target base.
    """
    gs = belief.prior.grid_size
    target = _target_base(belief, planner, params)

    # Resolve the leash (centre, radius), by precedence.
    leash_centre, leash_radius = None, None
    if params.camp_leash is not None:
        leash_centre, leash_radius = belief.prior.our_base, params.camp_leash
    elif target is not None:
        base, eff_hp, bombs_needed = target
        if eff_hp <= params.soften_floor:
            leash_centre, leash_radius = base, bombs_needed + 1

    best, best_tile = -INF, None
    for cell, value in belief.remaining_collectibles().items():
        if (leash_centre is not None
                and chebyshev(cell, leash_centre) > leash_radius):
            continue
        d = planner.dist_to(cell)
        # d == 0: already on the tile (first_action would be None); d == INF: unreachable.
        if d == INF or d == 0:
            continue
        score = value / (1.0 + d)
        if target is not None:
            near = chebyshev(cell, target[0])
            score += params.sweep_base_gradient * (1.0 - near / gs)
        if score > best:
            best, best_tile = score, cell
    if best_tile is None:
        return None
    return planner.first_action(best_tile)


def default(belief, danger, planner, params):
    """Advance toward the chosen enemy base. Yields (None) when no live,
    non-doomed base exists or the chosen base is unreachable from here."""
    target = _target_base(belief, planner, params)
    if target is None:
        return None
    return planner.first_action(target[0])


def camp(belief, danger, planner, params):
    """Defensive homebody. Territory is `camp_leash` Chebyshev of our base.

    Bomb enemies inside the territory; return home when outside the leash;
    otherwise None, letting forage/sweep collect within the leash.

    Precondition: ``params.camp_leash is not None`` — this layer is only used
    by the ``camper`` strategy, which sets ``camp_leash=4``.
    """
    leash = params.camp_leash
    base = belief.prior.our_base
    loc = belief.location

    # Defend: an enemy inside our territory.
    if belief.team_bombs > 0:
        threats = [e for e in belief.enemies if chebyshev(e, base) <= leash]
        if threats:
            if any(bomb_reaches(loc, e, belief) for e in threats):
                return PLACE_BOMB
            target = min(threats, key=lambda e: planner.dist_to(e))
            gs = belief.prior.grid_size
            best, best_tile = INF, None
            for x in range(gs):
                for y in range(gs):
                    if not bomb_reaches((x, y), target, belief):
                        continue
                    d = planner.dist_to((x, y))
                    if d < best:
                        best, best_tile = d, (x, y)
            if best_tile is not None and best != INF:
                action = planner.first_action(best_tile)
                if action is not None:
                    return action

    # Return home if outside the leash.
    if chebyshev(loc, base) > leash:
        return planner.first_action(base)

    return None


def hunt(belief, danger, planner, params):
    """Opportunistically bomb enemy agents already in blast range.

    Counts only LIVE enemies the bomb would actually hit (LOS + Chebyshev 2);
    frozen enemies are excluded. Our bomb is friendly-fire safe, so no escape
    route is needed. The bomb reserve rises by one while live enemy bases
    remain — bombs are kept for base kills:
      - live bases remain: 2+ enemies need >=2 bombs, a lone enemy >=3.
      - endgame (no live base): 2+ enemies need >=1, a lone enemy >=2.

    `danger`, `planner` are unused — kept for cascade-uniform layer typing.
    """
    if belief.team_bombs <= 0:
        return None
    loc = belief.location
    hits = sum(1 for e in belief.live_enemies() if bomb_reaches(loc, e, belief))
    if hits == 0:
        return None
    base_reserve = 1 if belief.live_enemy_bases() else 0
    min_bombs = (1 if hits >= 2 else 2) + base_reserve
    if belief.team_bombs >= min_bombs:
        return PLACE_BOMB
    return None


def hold(belief, danger, planner, params):
    """Final fallback for the camper — stay on station."""
    return STAY
