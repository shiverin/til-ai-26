"""The cascade runner: act() executes a Strategy's layer sequence."""
from scripted.danger import DangerMap
from scripted.geometry import BACKWARD, FORWARD, LEFT, PLACE_BOMB, RIGHT, STAY
from scripted.pathfind import build_planner
from scripted.strategies import STRATEGIES


def _first_legal(mask, preference):
    """First action in `preference` allowed by `mask`; if none match, any legal
    action in the mask; STAY only as a last resort."""
    for a in preference:
        if 0 <= a < len(mask) and mask[a] == 1:
            return a
    for a in range(len(mask)):
        if mask[a] == 1:
            return a
    return STAY


def _record(belief, action, layer):
    """Record the layer/source that produced `action` on the belief (read by
    the visualizer overlay; never affects behaviour), then return the action
    unchanged. A `PLACE_BOMB` is also logged to `belief.own_bombs` — the action
    mask guarantees the env will actually place it. `layer` is a cascade layer
    function's `__name__` (e.g. "survive", "sweep") or one of the literal
    strings "first_legal", "frozen".
    """
    if action == PLACE_BOMB:
        belief.record_own_bomb()
    belief.last_layer = layer
    return action


def act(belief, action_mask, strategy=None):
    """Run a Strategy's layer cascade and return a legal action int.

    `belief` must already be updated with the current observation.
    `strategy` defaults to the balanced strategy (the qualifier agent).
    """
    if strategy is None:
        strategy = STRATEGIES["balanced"]
    mask = list(action_mask)
    if belief.frozen_ticks > 0:
        return _record(belief, _first_legal(mask, [STAY]), "frozen")

    danger = DangerMap(belief.enemy_bombs, belief)
    planner = build_planner(belief, danger)

    for layer in strategy.layers:
        action = layer(belief, danger, planner, strategy.params)
        if action is not None and 0 <= action < len(mask) and mask[action] == 1:
            return _record(belief, action, layer.__name__)

    return _record(
        belief, _first_legal(mask, [FORWARD, BACKWARD, LEFT, RIGHT, STAY]),
        "first_legal")
