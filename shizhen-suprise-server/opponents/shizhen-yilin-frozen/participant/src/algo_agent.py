"""Fortified compound-interest economy agent (see PLAN.md).

Thin orchestrator implementing the per-turn pipeline (PLAN B3):

    0. safety shell: ~7.5s wall-clock budget + try/except per module
    1. ingest observation into persistent WorldMemory
    2. threat assessment
    3. diplomacy (accept all, propose to newly met, break with dead/campers)
    4. combat: focus-fire allocator
    5. economy: threat-responsive production, then expansion
    6. movement: kiting -> garrison -> scouts -> rallies
    7. validate every action against a local simulation

Pure algorithmic agent — no LLM, no network: the decisive layers (hex math,
range checks, focus fire) are what code is perfect at, and chat is treated
purely as a threat surface (only __system__ messages are ever read).
"""

from __future__ import annotations

import logging
import time

from agent_base import PlayerAgent
from engine.actions import ActionPayload

import combat
import diplomacy
import economy
import movement
import scouting
import threats
import validate
from world import WorldMemory

log = logging.getLogger("agent")

TURN_BUDGET_SECONDS = 7.5


class AlgoAgent(PlayerAgent):
    def __init__(self) -> None:
        self.world = WorldMemory()

    async def decide(self, observation: dict) -> ActionPayload:
        start = time.monotonic()
        pid = observation.get("player_id", "unknown")
        turn = observation.get("turn_number", 0)
        actions: list = []

        def over_budget() -> bool:
            return time.monotonic() - start > TURN_BUDGET_SECONDS

        try:
            self.world.ingest(observation)
        except Exception:
            log.exception("turn %s: ingest failed — no-op turn", turn)
            return ActionPayload(player_id=pid, turn_number=turn, actions=[])
        world = self.world

        try:
            threat = threats.assess(world)
        except Exception:
            log.exception("turn %s: threat assessment failed", turn)
            threat = threats.ThreatReport()

        war_prep = threat.war_prep

        try:
            actions += diplomacy.decide(world)
        except Exception:
            log.exception("turn %s: diplomacy failed", turn)

        engaged: dict = {}
        if not over_budget():
            try:
                attack_actions, engaged = combat.plan_attacks(world, threat)
                actions += attack_actions
            except Exception:
                log.exception("turn %s: combat failed", turn)

        reserved: set = set()
        if not over_budget():
            try:
                actions += economy.decide(world, threat, reserved)
            except Exception:
                log.exception("turn %s: economy failed", turn)

        if not over_budget():
            try:
                scout_goals = scouting.assign(world, war_prep)
            except Exception:
                log.exception("turn %s: scouting failed", turn)
                scout_goals = {}
            try:
                actions += movement.plan_moves(
                    world, threat, engaged, reserved, scout_goals
                )
            except Exception:
                log.exception("turn %s: movement failed", turn)

        try:
            actions = validate.validate(world, actions)
            world.reconcile_ledger(turn, actions)
        except Exception:
            log.exception("turn %s: validator failed — submitting unvalidated", turn)

        return ActionPayload(player_id=pid, turn_number=turn, actions=actions)
