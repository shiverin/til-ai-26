"""Manages the AE model — deterministic scripted agent (no neural network)."""

import os

from scripted import greedy
from scripted.belief import Belief, _scalar
from scripted.decide import act
from scripted.strategies import STRATEGIES
from scripted.map_prior import MapPrior


class AEManager:
    """Serves the scripted AE agent. One instance per server process."""

    GREEDY = "greedy"

    def __init__(self):
        self.prior = MapPrior.load()
        self.belief = Belief()
        self._episode_started = False   # True once a step-0 (round start) has been processed
        # Which scripted strategy to serve; set per-image via the AE_STRATEGY
        # env var (Docker build-arg). Defaults to the qualifier agent.
        name = os.environ.get("AE_STRATEGY", "balanced")
        self._greedy = name == self.GREEDY
        if self._greedy:
            self.strategy = None
        elif name in STRATEGIES:
            self.strategy = STRATEGIES[name]
        else:
            raise ValueError(
                f"AE_STRATEGY={name!r} is not a known strategy; "
                f"choose one of {sorted(list(STRATEGIES) + [self.GREEDY])}")

    def ae(self, observation: dict) -> int:
        """Return the next action for the agent.

        Args:
            observation: environment observation; see `ae/README.md`.

        Returns:
            An integer action in [0, 6).
        """
        # _scalar handles int / list / numpy-array forms uniformly.
        step = _scalar(observation["step"])

        # step == 0 marks a new round; the eval never calls /reset.
        if step == 0 or not self._episode_started:
            self.prior.identify_team(observation["base_location"])
            self.belief.reset(self.prior)
            self._episode_started = True

        self.belief.update(observation)
        if self._greedy:
            return greedy.act(self.belief, observation["action_mask"])
        return act(self.belief, observation["action_mask"], self.strategy)
