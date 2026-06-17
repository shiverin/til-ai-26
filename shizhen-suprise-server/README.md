# Shizhen SURPRISE Multi-Agent Test Server

This folder contains a standalone harness for running several `til-26-surprise`
participant agents in the same game. It reuses the canonical engine from:

```text
/home/jupyter/til-26-surprise/server/src
```

Each local agent folder is launched as its own existing `participant/src/server.py`
process on a separate port. The harness then calls every agent over `POST /observe`,
using the same HTTP contract as the competition, and writes a replay to
`./replays/`.

## One-Shot Run

From this folder:

```bash
python server.py run --config agents.example.json
```

Or pass agents directly:

```bash
python server.py run \
  --agent gpt=/home/jupyter/til-26-surprise-shizhen-gpt \
  --agent gemini=/home/jupyter/til-26-surprise-shizhen-gemini \
  --agent yilin=/home/jupyter/til-26-surprise-yilin \
  --max-turns 10000
```

The default turn cap is `10000`, which is intended as a safety limit while still
letting the match run until a single winner is eliminated through normal game
rules. If the cap is hit with multiple survivors, the result is
`turn_limit_reached`.

You can also use already-running agent servers:

```bash
python server.py run \
  --agent local-a=http://127.0.0.1:6700 \
  --agent local-b=http://127.0.0.1:6701
```

## API Server

Start the test server:

```bash
python server.py serve --host 127.0.0.1 --port 8088
```

Launch a match:

```bash
curl -sS -X POST http://127.0.0.1:8088/matches \
  -H 'content-type: application/json' \
  --data @agents.example.json
```

Poll status:

```bash
curl -sS http://127.0.0.1:8088/matches/<match_id>
```

Stop a running match:

```bash
curl -sS -X POST http://127.0.0.1:8088/matches/<match_id>/stop
```

## Config Shape

```json
{
  "seed": 67,
  "map_width": 35,
  "map_height": 30,
  "max_turns": 10000,
  "response_timeout": 10.0,
  "agents": [
    {"name": "agent-a", "path": "/path/to/til-26-surprise-copy"},
    {"name": "agent-b", "url": "http://127.0.0.1:6701"},
    {"name": "random-1", "kind": "random"}
  ]
}
```

For local folders, `path` can be the repo root, `participant/`, or
`participant/src/`. Use `"agent": "llm"` and an `env` object if you want to
start an LLM participant:

```json
{
  "name": "llm-agent",
  "path": "/home/jupyter/til-26-surprise",
  "agent": "llm",
  "env": {
    "OPENROUTER_API_KEY": "sk-..."
  }
}
```

Per-agent logs are written to `./logs/<match_id>/`, and replays are written to
`./replays/<match_id>.jsonl`.
