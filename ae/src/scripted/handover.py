"""Latching handover predicate for the hybrid post-opener controller.

Product B (post-opener): fires once the scripted opener has demonstrably done its
job — the first OBSERVED enemy-base destruction (`belief.dead_bases`, enemy-only)
— or at a step fallback for adverse seeds. The object is stateless; the caller
latches `handover_fired` on the first True. Product A (endgame forager) instead
uses `len(belief.live_enemy_bases()) == 0`.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class HandoverTrigger:
    min_destroyed_enemy_bases: int = 1
    step_fallback: int = 60

    def __call__(self, belief) -> bool:
        """Return True once the handover condition is met (latched by the caller)."""
        return (len(belief.dead_bases) >= self.min_destroyed_enemy_bases
                or belief.step >= self.step_fallback)
