"""DAgger behavior cloning of the `balanced` scripted teacher into the
SymbolicTransformerActor, then the BC gate.

Same pipeline as run_bc_large.py (200 pure-teacher episodes + 2 x 100 DAgger
rounds), but trains the transformer actor. Model scale is read from env vars
(TF_D_MODEL, TF_N_LAYERS, TF_N_HEADS, TF_FFN_DIM, TF_DROPOUT) so scale sweeps
need no code edit. Saves ae/src/policy_transformer_bc.pt as {state_dict, cfg}.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from bc import bc_gate, collect_dagger_dataset, train_bc
from policy import SymbolicTransformerActor

TEACHER = "balanced"


def _config_from_env():
    """Build the SymbolicTransformerActor config from TF_* env vars."""
    cfg = {
        "d_model": int(os.environ.get("TF_D_MODEL", 64)),
        "n_layers": int(os.environ.get("TF_N_LAYERS", 4)),
        "n_heads": int(os.environ.get("TF_N_HEADS", 4)),
        "dropout": float(os.environ.get("TF_DROPOUT", 0.1)),
    }
    ffn = os.environ.get("TF_FFN_DIM")
    if ffn is not None:
        cfg["ffn_dim"] = int(ffn)
    return cfg


def main():
    t0 = time.time()
    cfg = _config_from_env()
    print(f"transformer config: {cfg}", flush=True)
    actor = SymbolicTransformerActor(**cfg)

    # Round 1 — pure teacher (beta = 1.0).
    print("R1: collecting 200 pure-teacher episodes...", flush=True)
    ds = collect_dagger_dataset(TEACHER, None, 1.0, 200, list(range(200)))
    print(f"  R1 dataset: {len(ds)} samples  [{time.time() - t0:.0f}s]",
          flush=True)
    train_bc(actor, ds, epochs=20)
    print(f"  R1 trained  [{time.time() - t0:.0f}s]", flush=True)

    # Rounds 2-3 — DAgger aggregation with the partially-trained actor.
    for rnd in range(2):
        seeds = list(range(1000 + rnd * 1000, 1100 + rnd * 1000))
        print(f"R{rnd + 2}: collecting 100 DAgger episodes (beta=0.5)...",
              flush=True)
        more = collect_dagger_dataset(TEACHER, actor, 0.5, 100, seeds)
        ds += more
        print(f"  dataset now {len(ds)} samples  [{time.time() - t0:.0f}s]",
              flush=True)
        train_bc(actor, ds, epochs=20)
        print(f"  R{rnd + 2} trained  [{time.time() - t0:.0f}s]", flush=True)

    print(f"FINAL dataset: {len(ds)} samples", flush=True)

    passed, detail = bc_gate(actor, TEACHER)
    print(f"BC GATE {'PASS' if passed else 'FAIL'}: {detail}", flush=True)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "src", "policy_transformer_bc.pt")
    actor.save_checkpoint(out)
    print(f"saved {out}  [total {time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
