"""Predict which destructible walls an ally bomb's blast opens.

Faithful port of til_environment's bomb blast geometry (``supercover_line`` +
``_los_to_tile`` + ``_directional_blast`` Pass 2), reimplemented numpy-free
against the agent's own wall belief so it runs in the served container.

A bomb opens a destructible wall when the bomb has line-of-sight to at least
one of the two tiles that wall separates, within Chebyshev ``BLAST_RADIUS``.
"""
from scripted.geometry import MOVE

BLAST_RADIUS = 2


def supercover_line(start, end):
    """Tiles a straight line from `start` to `end` passes through.

    Vendored verbatim from til_environment.helpers.supercover_line.
    """
    x0, y0 = start
    x1, y1 = end
    dx, dy = x1 - x0, y1 - y0
    nx, ny = abs(dx), abs(dy)
    sign_x = 1 if dx > 0 else -1 if dx < 0 else 0
    sign_y = 1 if dy > 0 else -1 if dy < 0 else 0

    px, py = x0, y0
    tiles = [(px, py)]
    ix = iy = 0
    while ix < nx or iy < ny:
        if (1 + 2 * ix) * ny == (1 + 2 * iy) * nx:
            px += sign_x
            py += sign_y
            ix += 1
            iy += 1
        elif (1 + 2 * ix) * ny < (1 + 2 * iy) * nx:
            px += sign_x
            ix += 1
        else:
            py += sign_y
            iy += 1
        tiles.append((px, py))
    return tiles


def _wall(belief, tile, direction):
    """True if a wall (destructible or not) sits on `direction` side of `tile`."""
    dx, dy = MOVE[direction]
    return belief.is_wall(tile, (tile[0] + dx, tile[1] + dy))


def _los_to_tile(ox, oy, tx, ty, belief):
    """True if (tx, ty) has line-of-sight from (ox, oy) — no wall crossing.

    Vendored from til_environment.helpers._los_to_tile, with the env's raw
    wall-bit grid lookups replaced by `belief.is_wall`.
    """
    if tx == ox and ty == oy:
        return True
    path = supercover_line((ox, oy), (tx, ty))
    for i in range(len(path) - 1):
        cx, cy = path[i]
        nx, ny = path[i + 1]
        dx, dy = nx - cx, ny - cy
        if dx != 0 and dy != 0:
            h_dir = 0 if dx > 0 else 2          # RIGHT / LEFT
            v_dir = 1 if dy > 0 else 3          # DOWN / UP
            h_blocked = (_wall(belief, (cx, cy), h_dir)
                         or _wall(belief, (nx, cy), v_dir))
            v_blocked = (_wall(belief, (cx, cy), v_dir)
                         or _wall(belief, (cx, ny), h_dir))
            if h_blocked and v_blocked:
                return False
        else:
            d_val = 0 if dx == 1 else 1 if dy == 1 else 2 if dx == -1 else 3
            if _wall(belief, (cx, cy), d_val):
                return False
    return True


def walls_destroyed_by(bomb_cell, belief, blast_radius=BLAST_RADIUS):
    """Set of destructible wall pairs an ally bomb at `bomb_cell` opens.

    Pass 1 — collect blast cells: tiles within Chebyshev `blast_radius` that
    the bomb has line-of-sight to. Pass 2 — a destructible wall is opened if
    either tile it separates is a blast cell (and within radius). Mirrors
    til_environment `_directional_blast`.
    """
    ox, oy = bomb_cell
    gs = belief.prior.grid_size
    r = blast_radius

    reachable = set()
    for tx in range(max(0, ox - r), min(gs, ox + r + 1)):
        for ty in range(max(0, oy - r), min(gs, oy + r + 1)):
            if _los_to_tile(ox, oy, tx, ty, belief):
                reachable.add((tx, ty))

    destroyed = set()
    for pair, destructible in belief.prior.wall_between.items():
        if not destructible or pair in belief.destroyed_walls:
            continue
        a, b = tuple(pair)
        a_in = max(abs(a[0] - ox), abs(a[1] - oy)) <= r
        b_in = max(abs(b[0] - ox), abs(b[1] - oy)) <= r
        if not (a_in or b_in):
            continue
        if a in reachable or b in reachable:
            destroyed.add(pair)
    return destroyed


def bomb_reaches(bomb_cell, target, belief, blast_radius=BLAST_RADIUS):
    """True if a bomb placed at `bomb_cell` would hit `target`.

    Matches the env blast: the target must be within Chebyshev `blast_radius`
    AND have line-of-sight from the bomb (a wall in between stops the blast).
    """
    ox, oy = bomb_cell
    tx, ty = target
    if max(abs(tx - ox), abs(ty - oy)) > blast_radius:
        return False
    return _los_to_tile(ox, oy, tx, ty, belief)
