"""Episode-replay visualizer for AE self-play training.

Renders one labelled MP4 of an episode: each AEC slot agent_0..agent_5 is
drawn by the env, and a PIL legend band naming the policy controlling each
slot is stacked above every frame. Used both as a standalone CLI and by
train_selfplay.py's periodic viz hook. The vendored env is NOT modified.
"""
import os
import sys

# A plain `uv run python visualize.py` only has ae/training on sys.path; add
# ae/src so `from policy import ...` / `from features import ...` resolve.
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np

from evaluate import RandomAgent, ScriptedAgent, NeuralAgent
from policy import SymbolicTransformerActor
from scripted.strategies import STRATEGIES

import torch


def _spec_to_agent(spec):
    """Parse a per-slot spec string into an (agent, label) pair.

    Specs:
      - "random"          -> RandomAgent(), label "random"
      - "scripted:<name>" -> ScriptedAgent(<name>), label "scripted:<name>"
      - "ckpt:<path>"     -> NeuralAgent(SymbolicTransformerActor loaded from <path>),
                             label "ckpt:<basename>"
    Anything else raises ValueError.
    """
    if spec == "random":
        return RandomAgent(), "random"
    if spec.startswith("scripted:"):
        name = spec[len("scripted:"):]
        if name not in STRATEGIES:
            raise ValueError(
                f"unknown scripted strategy {name!r}; "
                f"valid: {sorted(STRATEGIES)}")
        return ScriptedAgent(name), f"scripted:{name}"
    if spec.startswith("ckpt:"):
        path = spec[len("ckpt:"):]
        if not os.path.exists(path):
            raise ValueError(f"checkpoint not found: {path}")
        actor = SymbolicTransformerActor.from_checkpoint(path)
        actor.eval()
        label = f"ckpt:{os.path.basename(path)}"
        return NeuralAgent(actor, name=label), label
    raise ValueError(
        f"bad agent spec {spec!r}; expected 'random', "
        f"'scripted:<name>', or 'ckpt:<path>'")


from PIL import Image, ImageDraw

from til_environment.renderer import _team_color

# Legend layout knobs.
LEGEND_ROW_H = 28          # KNOB: pixel height of one legend row
_SWATCH_PAD = 6            # inset of the color swatch within its row
_TEXT_X = LEGEND_ROW_H     # text starts just right of the swatch column


def _norm_color(c):
    """Normalize a _team_color() result to a (r, g, b) tuple of 0-255 ints.

    _team_color returns 0-255 ints for the fixed first-four-team palette and
    0-1 floats for golden-ratio-generated teams; normalize both to ints.
    """
    r, g, b = c
    if max(r, g, b) <= 1.0 and isinstance(r, float):
        return (int(r * 255), int(g * 255), int(b * 255))
    return (int(r), int(g), int(b))


def _build_legend(labels, width, layers=None):
    """Render the slot->policy legend as a uint8 RGB array (H, width, 3).

    One LEGEND_ROW_H-tall row per label: a team-color swatch (slot agent_K is
    team K in the novice 6-team layout) followed by 'agent_K = <label>'. When
    `layers` is given (one string per label), the agent's current cascade
    layer is appended to its row in brackets.
    """
    if layers is not None and len(layers) != len(labels):
        raise ValueError(
            f"layers length {len(layers)} != labels length {len(labels)}")
    height = LEGEND_ROW_H * len(labels)
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    for k, label in enumerate(labels):
        y0 = k * LEGEND_ROW_H
        color = _norm_color(_team_color(k))
        # color swatch
        draw.rectangle(
            [_SWATCH_PAD, y0 + _SWATCH_PAD,
             LEGEND_ROW_H - _SWATCH_PAD, y0 + LEGEND_ROW_H - _SWATCH_PAD],
            fill=color, outline=(0, 0, 0))
        # label text (default PIL bitmap font — no font file dependency)
        text = f"agent_{k} = {label}"
        if layers is not None and layers[k]:
            text += f"    [{layers[k]}]"
        draw.text((_TEXT_X, y0 + _SWATCH_PAD), text, fill=(0, 0, 0))
        # thin separator line under the row
        draw.line([0, y0 + LEGEND_ROW_H - 1, width, y0 + LEGEND_ROW_H - 1],
                  fill=(211, 211, 211))
    return np.asarray(img, dtype=np.uint8)


def _agent_layer(agent):
    """The agent's current cascade layer, or '' for agents with no cascade.

    Scripted agents expose it via `agent.belief.last_layer`; RandomAgent /
    NeuralAgent have no belief, so they resolve to ''."""
    belief = getattr(agent, "belief", None)
    return getattr(belief, "last_layer", None) or ""


import random

import imageio

from til_environment import bomberman_env
from til_environment.config import default_config

SLOTS = ["agent_0", "agent_1", "agent_2", "agent_3", "agent_4", "agent_5"]


def render_episode(slot_agents, out_path, *, fps=5, max_steps=200, seed=0):
    """Run one labelled render episode and write it to out_path as an MP4.

    slot_agents: {"agent_0": (agent, label), ..., "agent_5": (agent, label)}
        agent has .reset() and .action(observation) (evaluate.py adapters).
    Returns {"path": out_path, "scores": {slot: total_reward}, "steps": n}.
    """
    cfg = default_config()
    cfg.env.novice = True
    cfg.env.render_mode = "rgb_array"
    env = bomberman_env.basic_env(cfg=cfg, env_wrappers=[])

    labels = [slot_agents[s][1] for s in SLOTS]
    # the legend is rebuilt only when an agent's live cascade layer changes
    current_layers = {s: "" for s in SLOTS}
    legend = None
    last_layers = None

    random.seed(seed)
    env.reset(seed=seed)
    for agent, _ in slot_agents.values():
        agent.reset()

    totals = {s: 0.0 for s in SLOTS}
    frames = []
    steps = 0
    for slot in env.agent_iter():
        obs, reward, term, trunc, _ = env.last()
        totals[slot] += float(reward)
        if term or trunc:
            env.step(None)
            current_layers[slot] = ""        # drop the stale layer of a dead slot
        else:
            agent = slot_agents[slot][0]
            env.step(agent.action(obs))
            current_layers[slot] = _agent_layer(agent)
        frame = env.render()
        if frame is not None:
            frame = np.asarray(frame, dtype=np.uint8)
            new_layers = [current_layers[s] for s in SLOTS]
            if new_layers != last_layers:
                legend = _build_legend(labels, width=frame.shape[1],
                                       layers=new_layers)
                last_layers = new_layers
            frames.append(np.vstack([legend, frame]))
        steps += 1
        if steps >= max_steps * len(SLOTS):
            break
    env.close()

    if not frames:
        raise RuntimeError("render_episode captured no frames — is the env's render_mode set to 'rgb_array'?")
    imageio.mimwrite(out_path, frames, fps=fps, codec="libx264")
    # report episode length in env steps (agent_iter yields one turn per
    # slot, so a 6-agent env advances ~len(SLOTS) iters per env step).
    return {"path": out_path,
            "scores": totals,
            "steps": steps // len(SLOTS)}


from dataclasses import dataclass, field

VIZ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viz")


@dataclass
class VizArgs:
    """CLI args for a standalone replay render."""
    agents: list = field(default_factory=lambda: ["random"] * 6)  # 6 specs
    out: str = os.path.join(VIZ_DIR, "demo.mp4")
    fps: int = 5            # KNOB
    max_steps: int = 200    # KNOB
    seed: int = 0


def run_cli(args):
    """Build a slot_agents dict from args.agents and render one episode."""
    if len(args.agents) != len(SLOTS):
        raise ValueError(
            f"--agents needs exactly {len(SLOTS)} specs, "
            f"got {len(args.agents)}")
    slot_agents = {}
    for slot, spec in zip(SLOTS, args.agents):
        agent, label = _spec_to_agent(spec)
        slot_agents[slot] = (agent, label)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    return render_episode(slot_agents, args.out,
                          fps=args.fps, max_steps=args.max_steps,
                          seed=args.seed)


def main():
    import tyro
    args = tyro.cli(VizArgs)
    result = run_cli(args)
    print(f"wrote replay -> {result['path']} "
          f"({result['steps']} steps, scores={result['scores']})")


if __name__ == "__main__":
    main()
