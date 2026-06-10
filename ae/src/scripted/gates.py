"""Post-decision gates: opening rules and overrides that run after the layer
cascade picks an action.

A gate is `Callable[[belief, danger, planner, params, action], int | None]`.
Return `None` to pass the cascade's action through unchanged, or an int to
override (the override is dropped if it is illegal in the action mask). Gates
run in order; later gates see the result of earlier ones.

Gates can read belief state freely. They are *not* a substitute for layers —
use a layer when the rule depends on a routing/danger computation, use a gate
when the rule is a simple state-driven override (turn-0 opening, "stay sweep
until our bomb explodes", etc.).
"""
from scripted.belief import _trace_decision
from scripted.blast import bomb_reaches
from scripted.geometry import BACKWARD, FORWARD, MOVE, PLACE_BOMB, RIGHT
from scripted.layers import sweep, _base_doomed
from scripted.pathfind import BOMB_TIMER, build_planner


# Per-slot hard-coded opening books. Each entry is a list of action ints, one
# per tick from step 0. After the script runs out, the gate yields and the
# strategy cascade takes over. Keyed by team/slot index (prior.team).
_OPENING_BOOKS = {
    # 1: [PLACE_BOMB, BACKWARD, RIGHT, BACKWARD, FORWARD, FORWARD],
}


def force_turn0_bomb(belief, danger, planner, params, action):
    """Step 0 -> PLACE_BOMB unconditionally (mask permitting). Used to seed an
    opening breach that the agent then walks away from while it cooks."""
    if belief.step == 0:
        return PLACE_BOMB
    return None

def scripted_opening(belief, danger, planner, params, action):
    """Per-slot hard-coded opening book.

    Looks up the agent's slot via `belief.prior.team` (set by
    `MapPrior.identify_team` at step 0). If a book exists for that slot and
    `belief.step` is within its length, return the scripted action; otherwise
    yield (None) and let the strategy cascade run.

    Slot 1 currently uses [PLACE_BOMB, BACKWARD, RIGHT, BACKWARD, FORWARD,
    FORWARD] — a turn-0 breach plus a five-tick escape that clears the bomb's
    fuse window before the cascade resumes."""
    book = _OPENING_BOOKS.get(belief.prior.team)
    if book is None or belief.step >= len(book):
        return None
    return book[belief.step]


def _has_escape(belief, danger):
    """True if at least one tile reachable in <= BOMB_TIMER - 1 weighted ticks
    via a planner that drops a bomb on our tile NOW lies outside the bomb's
    blast — i.e., we can place a bomb and clear its blast before detonation."""
    breach = build_planner(belief, danger, place_bomb_first=True)
    gs = belief.prior.grid_size
    bomb_cell = tuple(belief.location)
    horizon = BOMB_TIMER - 1
    for x in range(gs):
        for y in range(gs):
            t = (x, y)
            if breach.dist_to(t) <= horizon and not bomb_reaches(bomb_cell, t, belief):
                return True
    return False


def body_block_resolve(belief, danger, planner, params, action):
    """Stuck-driven body-block resolver. Runs as a post-decision gate.

    When `belief.stuck_ticks >= params.stuck_trigger_ticks`:
      * Add the tile directly in front of us to belief.stuck_blacklist for
        params.stuck_blacklist_ttl ticks. The planner will treat this as a
        high-cost soft obstacle on subsequent ticks, so the cascade naturally
        pivots to an alternate base / corridor / forage objective.
      * If safe (escape verified, no self-bomb-blast hazard, not preempting
        survive), override the cascade pick with PLACE_BOMB. The bomb either
        kills/threatens the blocking agent or destroys a wall opening the way.

    Safety preconditions (any of these blocks the PLACE_BOMB override):
      - belief.last_layer == "survive" (survive's pick stands)
      - belief.own_bombs has a bomb whose blast covers belief.location
      - belief.team_bombs < 1 (nothing to drop)
      - escape verification fails
    """
    _trace_decision(belief, "body_block_resolve", "stuck_ticks", belief.stuck_ticks)
    if belief.stuck_ticks < params.stuck_trigger_ticks:
        return None

    # Blacklist the tile we actually tried to enter. stuck_ticks only
    # increments on intended moves (FORWARD/BACKWARD where the new location
    # didn't match `expected_location`), so `expected_location` is the cell
    # the move was blocked from entering — could be ahead OR behind. The
    # original `loc + facing` only matched the FORWARD case and mis-blamed
    # the wrong tile when a BACKWARD step got body-blocked. Fall back to the
    # forward tile if `expected_location` is unset or equal to `location`
    # (e.g. first tick / non-move actions).
    expected = belief.expected_location
    if expected is None or expected == belief.location:
        fx, fy = MOVE[belief.facing]
        expected = (belief.location[0] + fx, belief.location[1] + fy)
    belief.stuck_blacklist[expected] = belief.step + params.stuck_blacklist_ttl
    _trace_decision(belief, "body_block_resolve", "blacklisted", expected)

    # Don't override survive — it knows about danger we may not.
    if belief.last_layer == "survive":
        _trace_decision(belief, "body_block_resolve", "yield_survive_picked", True)
        return None

    # Don't bomb if one of our own bombs already covers our location.
    in_own_blast = any(bomb_reaches(cell, belief.location, belief)
                       for cell, _ in belief.own_bombs)
    if in_own_blast:
        _trace_decision(belief, "body_block_resolve", "yield_own_blast", True)
        return None

    if belief.team_bombs < 1:
        _trace_decision(belief, "body_block_resolve", "yield_no_bombs", True)
        return None

    if not _has_escape(belief, danger):
        _trace_decision(belief, "body_block_resolve", "yield_no_escape", True)
        return None

    _trace_decision(belief, "body_block_resolve", "place_bomb", True)
    return PLACE_BOMB


def sweep_while_own_bomb(belief, danger, planner, params, action):
    """While we have a bomb in-flight, override with `sweep`. Pairs with
    `force_turn0_bomb` so the agent grabs resources during its own bomb's fuse
    window instead of e.g. walking back into the blast (the cascade may pick
    survive's flee, which is also fine — but sweep banks reward)."""
    if not belief.own_bombs:
        return None
    return sweep(belief, danger, planner, params)


def strike_gate(belief, danger, planner, params, action):
    """Narrow tactical override: if the actor proposed a non-bomb action while in
    bomb range of a live, non-doomed enemy base (and we hold a bomb), place it.

    No own-bomb escape check — friendly fire is OFF (an agent takes zero damage
    from its own bombs), so a base hit carries no self-harm and an escape gate
    would only veto value. Range is the real `bomb_reaches` primitive
    (Chebyshev-2 + LOS), not literal adjacency. When several bases are hit the
    action is still just PLACE_BOMB; the hit set is trace-only diagnostics, not a
    target selection.
    """
    if action == PLACE_BOMB:
        return None                      # cascade already chose a bomb; nothing to add
    if belief.team_bombs < 1:
        return None
    targets = [base for base in belief.live_enemy_bases()
               if bomb_reaches(belief.location, base, belief)
               and not _base_doomed(belief, base)]
    if not targets:
        return None
    _trace_decision(belief, "strike_gate", "hit_bases", tuple(sorted(targets)))
    return PLACE_BOMB
