"""Time-aware planner over (tile, facing, phase) states.

Every action — turn LEFT/RIGHT, move FORWARD/BACKWARD, or STAY — costs one
game tick. A destructible wall is traversable only once a bomb has opened it
(at that bomb's detonation tick); an indestructible wall always blocks. A
state whose (cell, tick) an enemy blast covers is forbidden. Enemy bombs
detonate within BOMB_TIMER ticks and blasts are instantaneous, so past T_MAX
no hazard remains and no further wall opens — `phase` saturates at T_MAX and
the search collapses to an ordinary static Dijkstra tail.
"""
import heapq

from scripted.blast import walls_destroyed_by
from scripted.geometry import (
    BACKWARD, FORWARD, LEFT, MOVE, PLACE_BOMB, RIGHT, STAY,
)

BOMB_TIMER = 4                  # ticks from a bomb's placement to its detonation
T_MAX = BOMB_TIMER + 2          # past this tick no hazard remains; phase saturates
INF = float("inf")


class Planner:
    """Time-expanded search result: per-tile earliest arrival tick + routes."""

    def __init__(self, start, cost, prev):
        self._start = start
        self._prev = prev                       # state -> (prev_state, action)
        # Collapse the (tile, facing, phase) cost map to per-tile bests.
        tile_best, tile_goal = {}, {}
        for state, c in cost.items():
            tile = state[0]
            if c < tile_best.get(tile, INF):
                tile_best[tile], tile_goal[tile] = c, state
        self._tile_best = tile_best
        self._tile_goal = tile_goal

    def dist_to(self, tile):
        """Earliest arrival tick at `tile` in any facing (INF if unreachable)."""
        return self._tile_best.get(tuple(tile), INF)

    def steps_to(self, tile):
        """Action count of the cheapest route to `tile`. Every action costs one
        tick, so this equals dist_to; kept as a separate name for callers
        (survive) that read it as a step count."""
        return self.dist_to(tile)

    def first_action(self, tile):
        """First action of the earliest route to `tile`; None if unreachable
        or already there."""
        goal = self._tile_goal.get(tuple(tile))
        if goal is None or goal == self._start:
            return None
        s, action = goal, None
        while s != self._start:
            s, action = self._prev[s]
        return action


def _wall_open_ticks(belief, place_bomb_first):
    """Map each destructible wall pair to the earliest tick a known bomb opens
    it. Enemy and ally bombs open their walls at their remaining-timer tick.
    With `place_bomb_first`, a hypothetical bomb dropped on the agent's tile by
    the forced opening PLACE_BOMB (tick 1) opens its walls at 1 + BOMB_TIMER."""
    sources = list(belief.enemy_bombs.items()) + list(belief.ally_bombs.items())
    if place_bomb_first:
        sources.append((tuple(belief.location), 1 + BOMB_TIMER))
    opens = {}
    for cell, detonation in sources:
        for pair in walls_destroyed_by(cell, belief):
            if detonation < opens.get(pair, INF):
                opens[pair] = detonation
    return opens


def _edge_passable(belief, a, b, arrival_tick, open_ticks):
    """True if the agent may cross from tile a to adjacent tile b, arriving at
    `arrival_tick`. A destructible wall is crossable only at/after the tick a
    bomb opens it; an indestructible wall never is."""
    pair = frozenset({a, b})
    if pair in belief.destroyed_walls or pair not in belief.prior.wall_between:
        return True
    if not belief.prior.wall_between[pair]:              # indestructible
        return False
    return arrival_tick >= open_ticks.get(pair, INF)     # destructible


def build_planner(belief, danger, place_bomb_first=False):
    """Dijkstra over (tile, facing, phase) from the agent's current state.

    `phase` = min(tick, T_MAX); states at phase T_MAX form the hazard-free
    static tail. With `place_bomb_first`, the only action out of the start
    state is PLACE_BOMB — used to price the 'breach now' scenario.

    Costs (and the ticks `Planner.dist_to` / `steps_to` return) count actions
    taken from now, not absolute game-step numbers.
    """
    gs = belief.prior.grid_size
    open_ticks = _wall_open_ticks(belief, place_bomb_first)
    start = (tuple(belief.location), int(belief.facing), 0)
    cost = {start: 0}
    prev = {}
    pq = [(0, start)]
    while pq:
        c, state = heapq.heappop(pq)
        if c > cost.get(state, INF):
            continue
        tile, facing, phase = state
        nc = c + 1
        nphase = min(phase + 1, T_MAX)

        # `nc` / `nphase` are final for this iteration and `relax` is called
        # synchronously below — no late-binding closure hazard.
        def relax(ntile, nfacing, action):
            # An enemy blast covering (ntile, nc) is fatal — forbid the state.
            if danger.is_lethal_at(ntile, nc):
                return
            ns = (ntile, nfacing, nphase)
            if nc < cost.get(ns, INF):
                cost[ns] = nc
                prev[ns] = (state, action)
                heapq.heappush(pq, (nc, ns))

        if place_bomb_first and state == start:
            relax(tile, facing, PLACE_BOMB)         # forced opening move
            continue

        # Turns and STAY keep the agent on `tile`.
        relax(tile, (facing + 3) % 4, LEFT)
        relax(tile, (facing + 1) % 4, RIGHT)
        relax(tile, facing, STAY)
        # Forward / backward moves.
        for action, mdir in ((FORWARD, facing), (BACKWARD, (facing + 2) % 4)):
            dx, dy = MOVE[mdir]
            nb = (tile[0] + dx, tile[1] + dy)
            if not (0 <= nb[0] < gs and 0 <= nb[1] < gs):
                continue
            if nb in belief.frozen_enemies:
                continue
            if not _edge_passable(belief, tile, nb, nc, open_ticks):
                continue
            relax(nb, facing, action)
    return Planner(start, cost, prev)
