"""Multi-agent SURPRISE test server.

This harness reuses the canonical game engine from ``til-26-surprise/server/src``
and drives any number of participant HTTP servers in the same match. It can be
used as a one-shot CLI runner or as a small FastAPI service for launching and
monitoring matches.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parent
DEFAULT_ENGINE_ROOT = Path(
    os.environ.get("SURPRISE_ENGINE_ROOT", "/home/jupyter/til-26-surprise/server/src")
).expanduser()

if not (DEFAULT_ENGINE_ROOT / "game_runner.py").exists():
    raise SystemExit(
        "Cannot find the SURPRISE engine. Set SURPRISE_ENGINE_ROOT to the "
        "til-26-surprise/server/src directory."
    )

sys.path.insert(0, str(DEFAULT_ENGINE_ROOT.resolve()))

from engine.actions import ActionPayload, payload_from_dict  # noqa: E402
from game_runner import GameConfig, GameRunner, PlayerRegistration  # noqa: E402
from schemas.observation import build_observation  # noqa: E402


MAX_PLAYERS = 20
DEFAULT_MAX_TURNS = 10_000
DEFAULT_PORT_START = 7800
DEFAULT_RESPONSE_TIMEOUT = 10.0

log = logging.getLogger("surprise_multi_agent")


@dataclass
class AgentSpec:
    name: str
    kind: str = "http"
    path: str | None = None
    url: str | None = None
    agent: str = "algo"
    env: dict[str, str] = field(default_factory=dict)
    port: int | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AgentSpec":
        name = str(raw.get("name") or "").strip()
        if not name:
            raise ValueError("each agent needs a non-empty name")

        kind = str(raw.get("kind") or "").strip().lower()
        if not kind:
            kind = "external" if raw.get("url") else "http"
        if kind == "random":
            return cls(name=name, kind="random")

        env = raw.get("env") or {}
        if not isinstance(env, dict):
            raise ValueError(f"agent {name!r}: env must be an object")

        return cls(
            name=name,
            kind="external" if raw.get("url") else "http",
            path=str(raw["path"]) if raw.get("path") else None,
            url=str(raw["url"]).rstrip("/") if raw.get("url") else None,
            agent=str(raw.get("agent") or "algo"),
            env={str(k): str(v) for k, v in env.items()},
            port=int(raw["port"]) if raw.get("port") is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        out = {
            "name": self.name,
            "kind": self.kind,
            "agent": self.agent,
        }
        if self.path:
            out["path"] = self.path
        if self.url:
            out["url"] = self.url
        if self.port:
            out["port"] = self.port
        if self.env:
            out["env"] = {k: "***" if "KEY" in k or "TOKEN" in k else v for k, v in self.env.items()}
        return out


@dataclass
class MatchConfig:
    agents: list[AgentSpec]
    seed: int = 67
    map_width: int = 35
    map_height: int = 30
    max_turns: int = DEFAULT_MAX_TURNS
    response_timeout: float = DEFAULT_RESPONSE_TIMEOUT
    port_start: int = DEFAULT_PORT_START
    startup_timeout: float = 90.0
    replay_dir: str = str(ROOT / "replays")
    log_dir: str = str(ROOT / "logs")
    keep_agent_servers: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MatchConfig":
        agents = [AgentSpec.from_dict(a) for a in raw.get("agents", [])]
        random_count = int(raw.get("random_agents", 0) or 0)
        for i in range(random_count):
            agents.append(AgentSpec(name=f"random-{i + 1}", kind="random"))

        if len(agents) < 2:
            raise ValueError("provide at least two agents")
        if len(agents) > MAX_PLAYERS:
            raise ValueError(f"the engine supports at most {MAX_PLAYERS} players")

        names = [a.name for a in agents]
        if len(names) != len(set(names)):
            raise ValueError("agent names must be unique")

        max_turns = int(raw.get("max_turns", DEFAULT_MAX_TURNS) or DEFAULT_MAX_TURNS)
        if max_turns <= 0:
            max_turns = DEFAULT_MAX_TURNS

        return cls(
            agents=agents,
            seed=int(raw.get("seed", 67)),
            map_width=int(raw.get("map_width", 35)),
            map_height=int(raw.get("map_height", 30)),
            max_turns=max_turns,
            response_timeout=float(raw.get("response_timeout", DEFAULT_RESPONSE_TIMEOUT)),
            port_start=int(raw.get("port_start", DEFAULT_PORT_START)),
            startup_timeout=float(raw.get("startup_timeout", 90.0)),
            replay_dir=str(raw.get("replay_dir") or ROOT / "replays"),
            log_dir=str(raw.get("log_dir") or ROOT / "logs"),
            keep_agent_servers=bool(raw.get("keep_agent_servers", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "map_width": self.map_width,
            "map_height": self.map_height,
            "max_turns": self.max_turns,
            "response_timeout": self.response_timeout,
            "port_start": self.port_start,
            "startup_timeout": self.startup_timeout,
            "replay_dir": self.replay_dir,
            "log_dir": self.log_dir,
            "keep_agent_servers": self.keep_agent_servers,
            "agents": [a.to_dict() for a in self.agents],
        }


class HttpActor:
    def __init__(self, name: str, url: str, timeout: float) -> None:
        self.name = name
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.calls = 0
        self.errors = 0
        self.last_error: str | None = None
        self._client: httpx.AsyncClient | None = None

    async def open(self) -> None:
        self._client = httpx.AsyncClient(timeout=self.timeout)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def decide(self, observation: dict[str, Any]) -> ActionPayload | None:
        self.calls += 1
        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        close_after = self._client is None
        try:
            response = await client.post(f"{self.url}/observe", json=observation)
            response.raise_for_status()
            self.last_error = None
            return payload_from_dict(response.json())
        except Exception as exc:  # noqa: BLE001
            self.errors += 1
            self.last_error = str(exc)
            log.warning("agent %s failed on turn %s: %s", self.name, observation.get("turn_number"), exc)
            return None
        finally:
            if close_after:
                await client.aclose()

    def stats(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "errors": self.errors,
            "last_error": self.last_error,
            "url": self.url,
        }


class RandomActor:
    def __init__(self, name: str) -> None:
        from baseline_random import RandomAgent

        self.name = name
        self.agent = RandomAgent()
        self.calls = 0
        self.errors = 0
        self.last_error: str | None = None

    async def open(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def decide(self, observation: dict[str, Any]) -> ActionPayload | None:
        self.calls += 1
        try:
            self.last_error = None
            return await self.agent.decide(observation)
        except Exception as exc:  # noqa: BLE001
            self.errors += 1
            self.last_error = str(exc)
            return None

    def stats(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "errors": self.errors,
            "last_error": self.last_error,
            "url": "local://random",
        }


class MultiAgentRunner(GameRunner):
    def __init__(self, registrations: list[PlayerRegistration], config: GameConfig, actors: dict[str, Any]) -> None:
        super().__init__(registrations, config)
        self.actors = actors

    async def _collect_actions(self, player_urls: dict[str, str]) -> dict[str, ActionPayload]:  # type: ignore[override]
        assert self.state is not None
        state = self.state
        alive = [pid for pid in player_urls if state.players[pid].alive]

        async def one(pid: str) -> tuple[str, ActionPayload]:
            obs = build_observation(
                state,
                pid,
                self.diplomacy,
                self.chat_log,
                self.config.max_turns,
            )
            payload = await self.actors[pid].decide(obs)
            if payload is None:
                payload = ActionPayload(player_id=pid, turn_number=state.turn_number, actions=[])
            return pid, payload

        return dict(await asyncio.gather(*[one(pid) for pid in alive]))


@dataclass
class AgentProcess:
    spec: AgentSpec
    player_id: str
    port: int
    process: subprocess.Popen
    log_path: Path
    log_handle: Any

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def terminate(self) -> None:
        if self.process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)
        with contextlib.suppress(Exception):
            self.log_handle.close()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.spec.name,
            "player_id": self.player_id,
            "port": self.port,
            "url": self.url,
            "pid": self.process.pid,
            "returncode": self.process.poll(),
            "log_path": str(self.log_path),
        }


@dataclass
class MatchHandle:
    match_id: str
    config: MatchConfig
    status: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    runner: MultiAgentRunner | None = None
    processes: list[AgentProcess] = field(default_factory=list)
    actor_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    task: asyncio.Task | None = None

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def stop(self) -> None:
        self.status = "stopping"
        self.touch()
        if self.runner is not None:
            self.runner.stop()
        for proc in self.processes:
            proc.terminate()

    def snapshot(self) -> dict[str, Any]:
        state = self.runner.state if self.runner and self.runner.state else None
        alive: list[dict[str, Any]] = []
        turn = 0
        if state is not None:
            turn = state.turn_number
            alive = [
                {"player_id": p.id, "name": p.name}
                for p in state.alive_players()
            ]

        return {
            "match_id": self.match_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "turn": turn,
            "alive": alive,
            "config": self.config.to_dict(),
            "processes": [p.to_dict() for p in self.processes],
            "actor_stats": self.actor_stats,
            "result": self.result,
            "error": self.error,
        }


def _slug(value: str) -> str:
    out = []
    for char in value.lower():
        if char.isalnum():
            out.append(char)
        elif char in ("-", "_", "."):
            out.append(char)
        else:
            out.append("-")
    return "".join(out).strip("-") or "agent"


def _resolve_participant_src(path: str) -> Path:
    base = Path(path).expanduser().resolve()
    candidates = [
        base,
        base / "participant" / "src",
        base / "src",
    ]
    for candidate in candidates:
        if (candidate / "server.py").exists():
            return candidate
    raise ValueError(
        f"could not find participant server.py under {base}; expected repo root, "
        "participant/, or participant/src/"
    )


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _next_free_port(start: int, reserved: set[int]) -> int:
    port = start
    while port in reserved or not _port_available(port):
        port += 1
    reserved.add(port)
    return port


async def _wait_healthy(name: str, url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient() as client:
        last_error = ""
        while time.monotonic() < deadline:
            try:
                response = await client.get(f"{url.rstrip('/')}/health", timeout=2.0)
                if response.status_code == 200:
                    return
                last_error = f"HTTP {response.status_code}"
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
            await asyncio.sleep(0.5)
    raise TimeoutError(f"agent {name!r} at {url} did not become healthy: {last_error}")


def _start_agent_process(
    spec: AgentSpec,
    player_id: str,
    port: int,
    log_dir: Path,
) -> AgentProcess:
    src_dir = _resolve_participant_src(spec.path or "")
    log_path = log_dir / f"{player_id}-{_slug(spec.name)}.log"
    log_handle = open(log_path, "w", encoding="utf-8")

    env = os.environ.copy()
    env.update(spec.env)
    env["PORT"] = str(port)
    env["PLAYER_ID"] = player_id
    env.setdefault("AGENT", spec.agent)
    env.setdefault("LOG_LEVEL", "INFO")

    process = subprocess.Popen(
        [sys.executable, "server.py"],
        cwd=src_dir,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return AgentProcess(
        spec=spec,
        player_id=player_id,
        port=port,
        process=process,
        log_path=log_path,
        log_handle=log_handle,
    )


def _player_summary(runner: MultiAgentRunner, actors: dict[str, Any]) -> list[dict[str, Any]]:
    state = runner.state
    if state is None:
        return []

    rows: list[dict[str, Any]] = []
    for pid, player in state.players.items():
        buildings = state.buildings_for(pid)
        units = state.units_for(pid)
        rows.append(
            {
                "player_id": pid,
                "name": player.name,
                "alive": player.alive,
                "resources": player.resources.to_dict(),
                "bases": state.count_bases(pid),
                "buildings": len(buildings),
                "units": len(units),
                "actor": actors[pid].stats(),
            }
        )
    return rows


async def run_match(config: MatchConfig, handle: MatchHandle | None = None) -> dict[str, Any]:
    match_id = handle.match_id if handle else str(uuid.uuid4())[:8]
    replay_dir = Path(config.replay_dir).expanduser().resolve()
    log_dir = Path(config.log_dir).expanduser().resolve() / match_id
    replay_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if handle:
        handle.status = "starting"
        handle.touch()

    reserved_ports: set[int] = set()
    processes: list[AgentProcess] = []
    actors: dict[str, Any] = {}
    registrations: list[PlayerRegistration] = []

    try:
        for index, spec in enumerate(config.agents):
            player_id = f"player-{index}"
            if spec.kind == "random":
                actors[player_id] = RandomActor(spec.name)
                registrations.append(PlayerRegistration(player_id, spec.name, "local://random"))
                continue

            if spec.url:
                url = spec.url.rstrip("/")
                actors[player_id] = HttpActor(spec.name, url, config.response_timeout)
                registrations.append(PlayerRegistration(player_id, spec.name, url))
                continue

            port = spec.port or _next_free_port(config.port_start + index, reserved_ports)
            process = _start_agent_process(spec, player_id, port, log_dir)
            processes.append(process)
            actors[player_id] = HttpActor(spec.name, process.url, config.response_timeout)
            registrations.append(PlayerRegistration(player_id, spec.name, process.url))

        if handle:
            handle.processes = processes
            handle.actor_stats = {pid: actor.stats() for pid, actor in actors.items()}
            handle.touch()

        for process in processes:
            if process.process.poll() is not None:
                raise RuntimeError(
                    f"agent {process.spec.name!r} exited before health-check; "
                    f"see {process.log_path}"
                )
            await _wait_healthy(process.spec.name, process.url, config.startup_timeout)

        for spec, reg in zip(config.agents, registrations, strict=True):
            if spec.url:
                await _wait_healthy(spec.name, reg.callback_url, config.startup_timeout)

        for actor in actors.values():
            await actor.open()

        replay_path = replay_dir / f"{match_id}.jsonl"
        game_config = GameConfig(
            seed=config.seed,
            map_width=config.map_width,
            map_height=config.map_height,
            max_turns=config.max_turns,
            response_timeout=config.response_timeout,
            replay_path=str(replay_path),
        )
        runner = MultiAgentRunner(registrations, game_config, actors)
        runner.initialise()

        if handle:
            handle.runner = runner
            handle.status = "running"
            handle.touch()

        await runner.run()

        if handle:
            handle.actor_stats = {pid: actor.stats() for pid, actor in actors.items()}
            handle.touch()

        state = runner.state
        assert state is not None
        alive = state.alive_players()
        if len(alive) == 1:
            status = "winner"
            winner = {"player_id": alive[0].id, "name": alive[0].name}
        elif len(alive) == 0:
            status = "no_survivors"
            winner = None
        elif handle and handle.status == "stopping":
            status = "stopped"
            winner = None
        else:
            status = "turn_limit_reached"
            winner = None

        result = {
            "match_id": match_id,
            "status": status,
            "winner": winner,
            "turn": state.turn_number,
            "max_turns": config.max_turns,
            "replay_path": str(replay_path),
            "players": _player_summary(runner, actors),
        }
        if handle:
            handle.result = result
            handle.status = "completed" if status != "stopped" else "stopped"
            handle.touch()
        return result
    except Exception as exc:
        if handle:
            handle.status = "failed"
            handle.error = str(exc)
            handle.touch()
        raise
    finally:
        for actor in actors.values():
            await actor.close()
        if not config.keep_agent_servers:
            for process in processes:
                process.terminate()


MATCHES: dict[str, MatchHandle] = {}


def create_app():
    from fastapi import FastAPI, HTTPException, Request

    app = FastAPI(title="Shizhen SURPRISE multi-agent test server")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "matches": len(MATCHES)}

    @app.post("/matches")
    async def create_match(request: Request) -> dict[str, Any]:
        try:
            payload = await request.json()
            config = MatchConfig.from_dict(payload)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        match_id = str(uuid.uuid4())[:8]
        handle = MatchHandle(match_id=match_id, config=config)
        MATCHES[match_id] = handle

        async def background() -> None:
            try:
                await run_match(config, handle)
            except Exception:
                log.exception("match %s failed", match_id)

        handle.task = asyncio.create_task(background())
        return handle.snapshot()

    @app.get("/matches")
    async def list_matches() -> dict[str, Any]:
        return {"matches": [m.snapshot() for m in MATCHES.values()]}

    @app.get("/matches/{match_id}")
    async def get_match(match_id: str) -> dict[str, Any]:
        handle = MATCHES.get(match_id)
        if handle is None:
            raise HTTPException(status_code=404, detail="match not found")
        return handle.snapshot()

    @app.post("/matches/{match_id}/stop")
    async def stop_match(match_id: str) -> dict[str, Any]:
        handle = MATCHES.get(match_id)
        if handle is None:
            raise HTTPException(status_code=404, detail="match not found")
        handle.stop()
        return handle.snapshot()

    return app


app = create_app()


def _load_config_file(path: str) -> dict[str, Any]:
    with open(Path(path).expanduser(), encoding="utf-8") as file:
        return json.load(file)


def _parse_agent_arg(value: str) -> AgentSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("use NAME=PATH or NAME=http://host:port")
    name, target = value.split("=", 1)
    if target.startswith("http://") or target.startswith("https://"):
        return AgentSpec(name=name, kind="external", url=target.rstrip("/"))
    return AgentSpec(name=name, kind="http", path=target)


def _build_cli_config(args: argparse.Namespace) -> MatchConfig:
    if args.config:
        raw = _load_config_file(args.config)
    else:
        raw = {"agents": []}

    agents = [AgentSpec.from_dict(a) for a in raw.get("agents", [])]
    agents.extend(args.agent or [])
    agents.extend(AgentSpec(name=f"random-{i + 1}", kind="random") for i in range(args.random))

    raw.update(
        {
            "agents": [a.to_dict() for a in agents],
            "seed": args.seed if args.seed is not None else raw.get("seed", 67),
            "map_width": args.map_width if args.map_width is not None else raw.get("map_width", 35),
            "map_height": args.map_height if args.map_height is not None else raw.get("map_height", 30),
            "max_turns": args.max_turns if args.max_turns is not None else raw.get("max_turns", DEFAULT_MAX_TURNS),
            "response_timeout": args.response_timeout
            if args.response_timeout is not None
            else raw.get("response_timeout", DEFAULT_RESPONSE_TIMEOUT),
            "port_start": args.port_start if args.port_start is not None else raw.get("port_start", DEFAULT_PORT_START),
            "keep_agent_servers": args.keep_agent_servers or raw.get("keep_agent_servers", False),
        }
    )
    return MatchConfig.from_dict(raw)


async def _run_cli(args: argparse.Namespace) -> int:
    config = _build_cli_config(args)
    result = await run_match(config)
    print(json.dumps(result, indent=2))
    if result["status"] == "winner":
        print(f"\nWinner: {result['winner']['name']} ({result['winner']['player_id']}) at turn {result['turn']}")
        return 0
    print(f"\nFinished without a single winner: {result['status']} at turn {result['turn']}")
    return 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run multiple SURPRISE agents in one local match.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="run one match and exit")
    run_parser.add_argument("--config", help="JSON match config")
    run_parser.add_argument("--agent", action="append", type=_parse_agent_arg, help="agent as NAME=PATH or NAME=URL")
    run_parser.add_argument("--random", type=int, default=0, help="add this many in-process random agents")
    run_parser.add_argument("--seed", type=int)
    run_parser.add_argument("--map-width", type=int)
    run_parser.add_argument("--map-height", type=int)
    run_parser.add_argument("--max-turns", type=int)
    run_parser.add_argument("--response-timeout", type=float)
    run_parser.add_argument("--port-start", type=int)
    run_parser.add_argument("--keep-agent-servers", action="store_true")

    serve_parser = sub.add_parser("serve", help="start the FastAPI test server")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8088)

    args = parser.parse_args(argv)
    if args.command == "run":
        return asyncio.run(_run_cli(args))
    if args.command == "serve":
        import uvicorn

        uvicorn.run("server:app", host=args.host, port=args.port, reload=False)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
