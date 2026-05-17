"""In-process evaluation harness for the NLP RAG QA pipeline.

Loads the pipeline and the ModernBERT answer-equivalence scorer, runs the
local question set, and reports equiv_rate (overall + L1/L2) and retrieval
recall@k against the known source documents.

Usage:
  python eval_local.py            # full 883-question set
  python eval_local.py --limit 150  # quick subset for tuning iterations
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
# test_nlp.py and its AnswerEquivalenceEvaluator live in the repo test/ folder.
sys.path.insert(0, "/home/jupyter/til-ai-26/test")

import nlp_manager

DATA_DIR = "/home/jupyter/novice/nlp"
EVAL_MODEL = "/home/jupyter/til-ai-26/test/models/nlp_eval_512"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N questions")
    args = ap.parse_args()

    # Use the host model cache; the container uses /app/models.
    nlp_manager.MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

    # Load documents in sorted DOC-id order so corpus index <-> DOC-id is known.
    doc_paths = sorted(glob.glob(f"{DATA_DIR}/documents/*.txt"))
    doc_ids = [os.path.basename(p)[:-4] for p in doc_paths]
    documents = [open(p).read() for p in doc_paths]

    rows = [json.loads(ln) for ln in open(f"{DATA_DIR}/nlp.jsonl") if ln.strip()]
    if args.limit:
        rows = rows[:args.limit]

    print(f"Loading pipeline; indexing {len(documents)} documents ...",
          flush=True)
    manager = nlp_manager.NLPManager()
    manager.load_corpus(documents)

    from test_nlp import AnswerEquivalenceEvaluator
    scorer = AnswerEquivalenceEvaluator(model_path=EVAL_MODEL, max_length=512)

    preds, triples = [], []
    recall_hits = {1: 0, 3: 0, 5: 0}
    for i, r in enumerate(rows):
        q = r["question"]
        # retrieval recall: which docs do the reranked chunks come from
        chunks = manager.retriever.retrieve(q)
        retrieved_docs = []
        for c in chunks:
            d = doc_ids[c.doc_index]
            if d not in retrieved_docs:
                retrieved_docs.append(d)
        src = r["source_docs"][0]
        for k in recall_hits:
            if src in retrieved_docs[:k]:
                recall_hits[k] += 1
        pred = manager.qa(q)
        preds.append(pred)
        triples.append((q, r["answer"] or "", pred))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(rows)} answered", flush=True)

    results = scorer.batch_evaluate(triples)
    by_diff = {"L1": [0, 0], "L2": [0, 0]}
    for r, res in zip(rows, results):
        d = r["difficulty"]
        by_diff.setdefault(d, [0, 0])
        by_diff[d][1] += 1
        by_diff[d][0] += int(res.equivalent)
    n = len(rows)
    equiv = sum(res.equivalent for res in results)

    print("\n==== RESULTS ====")
    print(f"equiv_rate (score): {equiv / n:.4f}  ({equiv}/{n})")
    for d, (hit, tot) in sorted(by_diff.items()):
        if tot:
            print(f"  {d}: {hit / tot:.4f}  ({hit}/{tot})")
    for k, hit in sorted(recall_hits.items()):
        print(f"retrieval recall@{k}: {hit / n:.4f}")

    out = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(out, "w") as f:
        json.dump({"equiv_rate": equiv / n, "preds": preds}, f, indent=2)
    print(f"\nSaved predictions to {out}")


if __name__ == "__main__":
    main()
