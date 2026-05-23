"""Bench: transcribe N novice clips once, score 3 postprocess configs + per-rule deltas.

Usage:
    cd /home/jupyter/til-ai-26/asr && python bench_postprocess.py --limit 200

Writes /home/jupyter/til-ai-26/asr/bench_postprocess_results.json.
"""

import argparse
import json
import os
import sys

# Make `postprocess` importable (must come first so it shadows asr_shizhen's
# postprocess package when both dirs are on sys.path).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Make `bench.scoring` from asr_shizhen importable (inserted at position 1 so
# the local postprocess/ above stays at position 0 and wins the import).
sys.path.insert(1, "/home/jupyter/til-ai-26/asr_shizhen")

import nemo.collections.asr as nemo_asr

from bench.scoring import score_corpus
from postprocess import spelling_norm
from postprocess.pipeline import make_pipeline

_MODEL_PATH = "/home/jupyter/til-ai-26/asr/models/parakeet_finetuned.nemo"
_SAMPLES_DIR = "/home/jupyter/novice/asr"
_JSONL = os.path.join(_SAMPLES_DIR, "asr.jsonl")
_RESULTS_PATH = "/home/jupyter/til-ai-26/asr/bench_postprocess_results.json"


def load_refs(limit):
    refs = []
    with open(_JSONL) as f:
        for line in f:
            r = json.loads(line)
            r["path"] = os.path.join(_SAMPLES_DIR, r["audio"])
            refs.append(r)
            if len(refs) >= limit:
                break
    return refs


def transcribe_all(paths, batch_size):
    print(f"Loading model from {_MODEL_PATH}...", flush=True)
    model = nemo_asr.models.ASRModel.restore_from(_MODEL_PATH).eval().cuda()
    print(f"Transcribing {len(paths)} clips (batch_size={batch_size})...",
          flush=True)
    outputs = model.transcribe(paths, batch_size=batch_size)
    return [(getattr(o, "text", o) or "").strip() for o in outputs]


def score_config(name, enabled, raw, references):
    pipe = make_pipeline(enabled)
    hyps = [pipe(t) for t in raw]
    s = score_corpus(references, hyps)
    print(f"  {name:25s}  WER={s['wer']:.4f}  score={s['score']:.4f}",
          flush=True)
    return s, hyps


def per_rule_analysis(raw_after_numbers_disfluency, references, baseline_wer):
    """For each spelling rule, score what changes when ONLY it is applied
    on top of the numbers+disfluency config. Report fire-count + WER delta."""
    print("\n=== Per-spelling-rule deltas "
          "(applied on top of numbers+disfluency) ===", flush=True)
    per_rule = {}
    for be, ae in spelling_norm.RULES.items():
        applied = [
            spelling_norm.apply_rules(t, {be: ae})
            for t in raw_after_numbers_disfluency
        ]
        fired = sum(
            1 for a, b in zip(raw_after_numbers_disfluency, applied) if a != b)
        if fired == 0:
            per_rule[be] = {"to": ae, "fired_clips": 0, "wer_delta": 0.0}
            continue
        s = score_corpus(references, applied)
        delta = s["wer"] - baseline_wer
        per_rule[be] = {
            "to": ae, "fired_clips": fired, "wer_delta": delta}
        sign = "+" if delta >= 0 else ""
        print(f"  {be:14s} -> {ae:14s}  fired={fired:3d}  "
              f"wer_delta={sign}{delta:.5f}", flush=True)
    return per_rule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="GPU batch size. Default 4 keeps peak VRAM "
                             "low for shared GPUs; raise to 16 when alone.")
    args = parser.parse_args()

    refs = load_refs(args.limit)
    if not refs:
        raise SystemExit(f"No refs found in {_JSONL}.")
    paths = [r["path"] for r in refs]
    references = [r["transcript"] for r in refs]

    raw = transcribe_all(paths, args.batch_size)

    print("\n=== Config scores ===", flush=True)
    configs = {
        "baseline": [],
        "numbers_disfluency": ["numbers", "disfluency"],
        "full_pp": ["numbers", "spelling_norm", "disfluency"],
    }
    results = {}
    hyps_by_config = {}
    for name, enabled in configs.items():
        s, h = score_config(name, enabled, raw, references)
        results[name] = s
        hyps_by_config[name] = h

    per_rule = per_rule_analysis(
        hyps_by_config["numbers_disfluency"], references,
        results["numbers_disfluency"]["wer"])

    out = {
        "n_clips": len(refs),
        "configs": results,
        "per_rule": per_rule,
    }
    with open(_RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote results to {_RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
