"""Treaty state machine (PLAN A6 / C-Diplomacy).

Accept every incoming proposal; propose to every newly-met living player; break
treaties with eliminated partners (their inert buildings stay unattackable
otherwise). All diplomacy stops at the turn-200 cutoff — the engine voids
everything then.

Deliberately NO proximity/camping-based breaks: a treaty partner's units are
mechanically harmless (their attacks are invalid no-ops), and self-play showed
breaking over neighbours' loitering garrisons produces endless 5-turn war
cycles that grind both sides down — the exact wars the co-win condition says
to avoid.
"""

from __future__ import annotations

from engine.actions import BreakTreatyAction, ProposeTreatyAction, RespondTreatyAction
from engine.constants import TREATY_CUTOFF_TURN

from world import WorldMemory

PROPOSAL_COOLDOWN = 25  # turns between re-proposals to a non-partner


def decide(world: WorldMemory) -> list:
    if world.turn >= TREATY_CUTOFF_TURN or world.grid is None:
        return []
    actions: list = []

    # accept everything incoming (A6: peace makes their attacks invalid no-ops)
    for prop in world.incoming_proposals:
        actions.append(
            RespondTreatyAction(
                proposing_player_id=prop["proposer_id"],
                treaty_type=prop.get("treaty_type", "peace"),
                accept=True,
            )
        )

    # break only with the dead (frees their inert buildings for cleanup)
    for pid, treaty in world.treaties.items():
        if treaty.get("breaking_in_turns") is not None:
            continue  # already breaking
        if pid in world.eliminated:
            actions.append(
                BreakTreatyAction(
                    partner_player_id=pid,
                    treaty_type=treaty.get("treaty_type", "peace"),
                )
            )

    # propose to every met, living, non-partner player (with a re-try cooldown)
    proposed_this_turn = {p["proposer_id"] for p in world.incoming_proposals}
    for pid in world.known_players:
        if (
            pid in world.treaties
            or pid in world.eliminated
            or pid in proposed_this_turn
            or world.turn - world.proposals_sent.get(pid, -10**9) < PROPOSAL_COOLDOWN
        ):
            continue
        actions.append(ProposeTreatyAction(target_player_id=pid, treaty_type="peace"))
        world.proposals_sent[pid] = world.turn
    return actions
