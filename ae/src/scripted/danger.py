"""Bomb danger map. Conservative Chebyshev-radius-2 model (no line-of-sight)."""

BLAST_RADIUS = 2
_SAFE = 999  # sentinel meaning "never in blast range"


class DangerMap:
    """Per-cell soonest-detonation tick, bomb-overlap count, and the set of
    ticks each cell is hit by a blast — all from believed bombs."""

    def __init__(self, bombs, grid_size):
        """bombs: dict (x,y) -> timer ticks remaining."""
        self.grid_size = grid_size
        self._tick = {}     # (x,y) -> soonest detonation tick
        self._count = {}    # (x,y) -> number of bombs whose blast covers it
        self._lethal = {}   # (x,y) -> set of ticks a blast covers it
        for (bx, by), timer in bombs.items():
            for x in range(max(0, bx - BLAST_RADIUS),
                            min(grid_size, bx + BLAST_RADIUS + 1)):
                for y in range(max(0, by - BLAST_RADIUS),
                                min(grid_size, by + BLAST_RADIUS + 1)):
                    prev = self._tick.get((x, y), _SAFE)
                    self._tick[(x, y)] = min(prev, timer)
                    self._count[(x, y)] = self._count.get((x, y), 0) + 1
                    self._lethal.setdefault((x, y), set()).add(timer)

    def ticks_to_danger(self, cell):
        """Soonest tick this cell is hit by a blast, or 999 if never."""
        return self._tick.get(tuple(cell), _SAFE)

    def overlap(self, cell):
        """Number of bombs whose blast covers `cell` (0 if safe)."""
        return self._count.get(tuple(cell), 0)

    def is_dangerous(self, cell, within=_SAFE - 1):
        """True if `cell` is hit by a blast at or before tick `within`."""
        t = self.ticks_to_danger(cell)
        return t != _SAFE and t <= within

    def is_lethal_at(self, cell, tick):
        """True if a blast covers `cell` at game tick `tick`.

        Blasts are instantaneous in til_environment (no lingering explosion),
        so a cell is lethal only on a bomb's exact detonation tick — which
        equals that bomb's remaining timer."""
        return tick in self._lethal.get(tuple(cell), ())
