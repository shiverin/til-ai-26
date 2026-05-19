"""Per-episode belief: the agent's internal mutable map.

Reset when observation["step"] == 0. Tracks walls (monotonic — only ever
opened), collectibles (simple re-check on sight), bombs, enemy agents, and
enemy-base liveness + health (a destroyed base never respawns).

Bombs are split by team. `enemy_bombs` feeds the danger model. `ally_bombs`
(our own) are tracked separately — a bomb deals no damage to its own team
(til_environment friendly-fire rule), so an ally bomb is never a danger, but
it still reveals which destructible walls are about to open.
"""
import numpy as np

from scripted.blast import walls_destroyed_by
from scripted.geometry import view_to_world
from scripted.pathfind import BOMB_TIMER

# Viewcone channel indices — mirror til_environment.observation.ViewChannel.
# test_scripted_channels.py asserts these stay in sync.
CH = {
    "WALL_RIGHT": 1, "WALL_DOWN": 2, "WALL_LEFT": 3, "WALL_UP": 4,
    "TILE_EMPTY": 5, "TILE_RECON": 6, "TILE_MISSION": 7, "TILE_RESOURCE": 8,
    "ENEMY_AGENT": 10, "ENEMY_AGENT_HEALTH": 22,
    "ENEMY_BASE": 12, "ENEMY_BASE_HEALTH": 24,
    "ALLY_BOMB": 17, "ENEMY_BOMB": 18,
    "ALLY_BOMB_TIMER": 19, "ENEMY_BOMB_TIMER": 20,
}
# Viewcone vision config (til_environment default: left2/right2/behind2/ahead4).
VC_BEHIND, VC_LEFT = 2, 2


def _scalar(x):
    """Extract an int from a scalar / length-1 list / tuple / array."""
    return int(np.asarray(x).flat[0])


class Belief:
    """Mutable per-episode world model. One instance reused across episodes."""

    def __init__(self):
        self.prior = None
        self.destroyed_walls = set()    # frozenset pairs known opened
        self.collected = set()          # (x,y) collectibles seen taken
        self.enemy_bombs = {}           # (x,y) -> timer; enemy bombs — the danger model
        self.ally_bombs = {}            # (x,y) -> timer; our own bombs (harmless to us)
        self.own_bombs = []             # [(cell, timer)]; bombs the agent placed
        self.enemies = set()            # (x,y) enemy cells visible this tick (cleared each update)
        self.frozen_enemies = set()     # (x,y) visible enemies at 0 health — motionless obstacles
        self.dead_bases = set()         # enemy base (x,y) seen destroyed — permanent, no respawn
        self.enemy_base_health = {}     # enemy base (x,y) -> last-seen health ratio [0,1]
        self._enemy_base_set = set()    # enemy base coords (set on reset, for O(1) lookup)
        self.location = None
        self.facing = 0
        self.team_bombs = 0
        self.step = 0
        self.frozen_ticks = 0
        self.health = 0.0
        self.base_health = 0.0          # our base's raw HP (not a ratio); 0 == destroyed
        # Debug instrumentation: name of the cascade layer / source act()
        # last fired ("survive", "sweep", "first_legal", …). Read by
        # the visualizer overlay; never affects behaviour.
        self.last_layer = None

    def reset(self, prior):
        """Start a new episode. `prior` is a team-identified MapPrior."""
        self.prior = prior
        self.destroyed_walls = set()
        self.collected = set()
        self.enemy_bombs = {}
        self.ally_bombs = {}
        self.own_bombs = []
        self.enemies = set()
        self.frozen_enemies = set()
        self.dead_bases = set()
        self.enemy_base_health = {}
        self._enemy_base_set = set(prior.enemy_bases)
        self.last_layer = None

    def is_wall(self, a, b):
        """True if an (intact) wall separates adjacent tiles a and b."""
        pair = frozenset({tuple(a), tuple(b)})
        if pair in self.destroyed_walls:
            return False
        return pair in self.prior.wall_between

    def is_destructible(self, a, b):
        """True if the wall between a and b exists and is destructible."""
        pair = frozenset({tuple(a), tuple(b)})
        if pair in self.destroyed_walls:
            return False
        return self.prior.wall_between.get(pair, False)

    def remaining_collectibles(self):
        """(x,y) -> value for collectibles believed still present."""
        return {c: v for c, v in self.prior.collectibles.items()
                if c not in self.collected}

    def base_alive(self, base_cell):
        """True unless this enemy base has been seen destroyed (permanent —
        bases never respawn)."""
        return base_cell not in self.dead_bases

    def live_enemy_bases(self):
        """Enemy base coords not currently believed destroyed."""
        return [b for b in self.prior.enemy_bases if self.base_alive(b)]

    def live_enemies(self):
        """Visible enemy cells that are not frozen. Frozen enemies are
        motionless obstacles, not threats or bomb targets."""
        return self.enemies - self.frozen_enemies

    def _fold_viewcone(self, vc, to_world):
        """Fold one (H, W, 25) viewcone tensor into the belief.

        `to_world(i, j)` maps a view index to a world (x, y) tile. Cells that
        are off-grid, or not visible (no tile-type channel set), are skipped.
        """
        gs = self.prior.grid_size
        rows, cols, _ = vc.shape
        for i in range(rows):
            for j in range(cols):
                cell = vc[i, j]
                # A cell is visible iff exactly one tile-type channel is set.
                tile_sum = (cell[CH["TILE_EMPTY"]] + cell[CH["TILE_RECON"]]
                            + cell[CH["TILE_MISSION"]] + cell[CH["TILE_RESOURCE"]])
                if tile_sum < 0.5:
                    continue
                wx, wy = to_world(i, j)
                if not (0 <= wx < gs and 0 <= wy < gs):
                    continue
                w = (wx, wy)

                # Collectibles: empty => collected; present => restore.
                if w in self.prior.collectibles:
                    if cell[CH["TILE_EMPTY"]] > 0.5:
                        self.collected.add(w)
                    else:
                        self.collected.discard(w)

                # Walls: monotonic — only record disappearances.
                self._update_walls(w, cell, gs)

                # Bombs, split by team. Enemy bombs feed the danger model;
                # ally bombs (our own) are harmless to us but tracked too.
                if cell[CH["ALLY_BOMB"]] > 0.5:
                    t = max(1, int(round(cell[CH["ALLY_BOMB_TIMER"]])))
                    self.ally_bombs[w] = min(self.ally_bombs.get(w, 999), t)
                if cell[CH["ENEMY_BOMB"]] > 0.5:
                    t = max(1, int(round(cell[CH["ENEMY_BOMB_TIMER"]])))
                    self.enemy_bombs[w] = min(self.enemy_bombs.get(w, 999), t)

                # Enemy agents. ENEMY_AGENT_HEALTH is the health ratio; it is
                # 0 when the enemy is frozen (HP depleted — motionless on its
                # tile until it respawns). The `<= 0.0` guard also absorbs any
                # float32 round-off.
                if cell[CH["ENEMY_AGENT"]] > 0.5:
                    self.enemies.add(w)
                    if cell[CH["ENEMY_AGENT_HEALTH"]] <= 0.0:
                        self.frozen_enemies.add(w)

                # Enemy base liveness + health. A known base coord seen
                # WITHOUT the ENEMY_BASE channel has been destroyed — and a
                # destroyed base never respawns, so the record is permanent.
                if w in self._enemy_base_set:
                    if cell[CH["ENEMY_BASE"]] > 0.5:
                        self.enemy_base_health[w] = float(
                            cell[CH["ENEMY_BASE_HEALTH"]])
                    else:
                        self.dead_bases.add(w)

    def record_own_bomb(self):
        """Record a bomb the agent just placed on its current tile. Stacks are
        kept as distinct list entries so effective-HP accounting can count
        them — the cell-keyed `ally_bombs` cannot. Call only after `update()`
        has set `location` for the tick."""
        assert self.location is not None, "record_own_bomb before update()"
        self.own_bombs.append((self.location, BOMB_TIMER))

    def update(self, observation):
        """Fold one observation into the belief."""
        self.location = tuple(int(c) for c in observation["location"])
        self.facing = int(observation["direction"])
        self.team_bombs = _scalar(observation["team_bombs"])
        self.step = _scalar(observation["step"])
        self.frozen_ticks = _scalar(observation["frozen_ticks"])
        self.health = float(np.asarray(observation["health"]).flat[0])
        self.base_health = float(np.asarray(observation["base_health"]).flat[0])

        # Decrement remembered bombs; the viewcone refresh below re-asserts
        # any still visible.
        self.enemy_bombs = {c: t - 1 for c, t in self.enemy_bombs.items()
                            if t - 1 > 0}
        # Ally bombs: when a timer runs out the bomb has detonated — credit the
        # destructible walls its blast opened to our wall belief. All
        # detonations are scored against the pre-detonation wall state.
        aged_ally, detonated = {}, []
        for c, t in self.ally_bombs.items():
            if t - 1 > 0:
                aged_ally[c] = t - 1
            else:
                detonated.append(c)
        self.ally_bombs = aged_ally
        opened = set()
        for c in detonated:
            opened |= walls_destroyed_by(c, self)
        self.destroyed_walls |= opened
        # Own bombs (action-sourced): age each by a tick, drop the detonated.
        self.own_bombs = [(c, t - 1) for c, t in self.own_bombs if t - 1 > 0]
        self.enemies = set()
        self.frozen_enemies = set()

        # Agent viewcone — a 7x5 cone rotated by the agent's facing. `loc` and
        # `facing` are bound from the fields set above so the mapping does not
        # depend on attribute-write ordering.
        agent_vc = np.asarray(observation["agent_viewcone"], dtype=np.float32)
        loc, facing = self.location, self.facing
        self._fold_viewcone(
            agent_vc,
            lambda i, j: view_to_world(loc, facing, (i - VC_BEHIND, j - VC_LEFT)))

        # Base viewcone — a square radius view centred on our base. A destroyed
        # base provides no vision (the env returns a degenerate all-zero
        # (1,1,25) view and a [0,0] base_location), so fold it only when the
        # base is alive.
        if self.base_health > 0.0:
            base_vc = np.asarray(observation["base_viewcone"], dtype=np.float32)
            bl = observation["base_location"]
            bx, by = int(bl[0]), int(bl[1])
            r = base_vc.shape[0] // 2          # square view: rows == cols == 2r+1
            self._fold_viewcone(base_vc, lambda i, j: (bx + i - r, by + j - r))

    def _update_walls(self, w, cell, gs):
        """Record any prior wall that the viewcone now shows as gone."""
        wx, wy = w
        for ch, (dx, dy) in (("WALL_RIGHT", (1, 0)), ("WALL_DOWN", (0, 1)),
                             ("WALL_LEFT", (-1, 0)), ("WALL_UP", (0, -1))):
            nx, ny = wx + dx, wy + dy
            if not (0 <= nx < gs and 0 <= ny < gs):
                continue
            pair = frozenset({w, (nx, ny)})
            if pair not in self.prior.wall_between or pair in self.destroyed_walls:
                continue
            if cell[CH[ch]] < 0.5:          # prior said wall here; viewcone says none
                self.destroyed_walls.add(pair)
