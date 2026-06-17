"""In-process test runner for AlgoAgent (PLAN D steps 5/8).

Modes:
    python local_test.py random [seed] [turns]    # vs 19 RandomAgents (stage-1 shape)
    python local_test.py self   [seed] [turns]    # SELF-PLAY: 20 copies of AlgoAgent

Runs the real GameRunner with all players in-process (no HTTP, no docker), prints
PASS/FAIL plus the plan's metrics: income curve, base count, units lost/killed.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "server" / "src"))
sys.path.insert(0, str(ROOT / "participant" / "src"))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

from engine.actions import ActionPayload  # noqa: E402
from game_runner import GameConfig, GameRunner, PlayerRegistration  # noqa: E402
from schemas.observation import build_observation  # noqa: E402

US = "player-0"


class InProcessRunner(GameRunner):
    def __init__(self, registrations, config, actors: dict) -> None:
        super().__init__(registrations, config)
        self.actors = actors
        self.metrics: list[dict] = []
        self.trace: list[str] = []
        self.slowest_turn = 0.0

    async def _collect_actions(self, player_urls):  # type: ignore[override]
        state = self.state
        alive = [pid for pid in player_urls if state.players[pid].alive]
        payloads = {}
        for pid in alive:
            obs = build_observation(
                state, pid, self.diplomacy, self.chat_log, self.config.max_turns
            )
            t0 = time.monotonic()
            try:
                payload = await self.actors[pid].decide(obs)
            except Exception as exc:  # mirror server.py's no-op shield
                print(f"  !! {pid} crashed on turn {state.turn_number}: {exc!r}")
                payload = None
            dt = time.monotonic() - t0
            if pid == US and dt > self.slowest_turn:
                self.slowest_turn = dt
            if payload is None:
                payload = ActionPayload(
                    player_id=pid, turn_number=state.turn_number, actions=[]
                )
            payloads[pid] = payload

        # per-turn trace for player-0 forensics
        us_p = state.players[US]
        bases = [
            e
            for e in state.entities.values()
            if e.owner_id == US and type(e).__name__ == "Base"
        ]
        units = [
            e
            for e in state.entities.values()
            if e.owner_id == US and not hasattr(e, "is_complete")
        ]
        near = {}
        for e in state.entities.values():
            if e.owner_id == US or hasattr(e, "is_complete"):
                continue
            for b in bases:
                if state.grid.distance(e.coord, b.coord) <= 6:
                    near[e.owner_id] = near.get(e.owner_id, 0) + 1
                    break
        from collections import Counter

        n_treaties = len(self.diplomacy.active_treaties_for(US))
        kinds = Counter(
            type(e).__name__ for e in state.entities.values() if e.owner_id == US
        )
        acts = Counter(
            a.type for a in (payloads[US].actions if US in payloads else [])
        )
        self.trace.append(
            f"t{state.turn_number:>3} gold={us_p.resources.to_dict().get('gold', 0):>6} "
            f"bases={[(str(b.coord), b.hp) for b in bases]} "
            f"have={dict(kinds)} acts={dict(acts)} "
            f"near={near} ntreaties={n_treaties}"
        )

        if state.turn_number % 25 == 0:
            print(
                f"  ..turn {state.turn_number} "
                f"alive={sum(1 for pl in state.players.values() if pl.alive)} "
                f"entities={len(state.entities)}",
                file=sys.stderr,
                flush=True,
            )
        if state.turn_number % 10 == 0:
            p = state.players[US]
            n_bases = state.count_bases(US)
            n_units = sum(
                1
                for e in state.entities.values()
                if e.owner_id == US and not hasattr(e, "is_complete")
            )
            self.metrics.append(
                {
                    "turn": state.turn_number,
                    "gold": p.resources.to_dict().get("gold", 0),
                    "bases": n_bases,
                    "units": n_units,
                    "alive": sum(1 for pl in state.players.values() if pl.alive),
                }
            )
        return payloads


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "random"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 67
    turns = int(sys.argv[3]) if len(sys.argv) > 3 else 300

    from algo_agent import AlgoAgent

    ids = [f"player-{i}" for i in range(20)]
    regs = [PlayerRegistration(pid, pid, "local://x") for pid in ids]
    actors: dict = {US: AlgoAgent()}
    if mode == "self":
        for pid in ids[1:]:
            actors[pid] = AlgoAgent()
    else:
        from baseline_random import RandomAgent

        for pid in ids[1:]:
            actors[pid] = RandomAgent()

    config = GameConfig(seed=seed, max_turns=turns)
    runner = InProcessRunner(regs, config, actors)
    runner.initialise()
    t0 = time.monotonic()
    asyncio.run(runner.run())
    wall = time.monotonic() - t0

    alive = runner.state.players[US].alive
    print("=" * 64)
    print(
        f"mode={mode} seed={seed}  RESULT: {'PASS' if alive else 'FAIL'} "
        f"(reached turn {runner.state.turn_number}/{turns}, {wall:.1f}s wall, "
        f"slowest our turn {runner.slowest_turn * 1000:.0f}ms)"
    )
    if mode == "self":
        survivors = [pid for pid, p in runner.state.players.items() if p.alive]
        print(f"self-play survivors: {len(survivors)}/20 -> {survivors}")
    from collections import Counter

    ours = Counter(
        type(e).__name__
        for e in runner.state.entities.values()
        if e.owner_id == US
    )
    print(f"our entities at end: {dict(ours)}")
    Path("trace_p0.log").write_text("\n".join(runner.trace))
    print("turn   gold  bases units alive")
    for m in runner.metrics:
        print(
            f"{m['turn']:>4} {m['gold']:>6} {m['bases']:>6} {m['units']:>5} {m['alive']:>5}"
        )


if __name__ == "__main__":
    main()
