"""Build RAFT (Retrieval-Augmented Fine-Tuning) training data.

For each training question we run the REAL retrieval pipeline (so the model
trains on exactly the kind of context — distractor chunks and all — it will see
at inference) and emit a chat example whose target is the gold answer in the
pipeline's output format.

Design choices:
  * No few-shot in the training prompt. Fine-tuning replaces the few-shot's
    job; dropping it also shortens the inference prompt later (a speed win).
  * Train/holdout split: holdout = first 150 questions (the existing eval
    subset, so post-RAFT eval stays comparable to the 0.761 baseline);
    train = the remaining 733.
  * Only retrieval models are loaded (embedder + reranker, ~2 GB) — vLLM is
    NOT needed here, so this runs alongside other GPU jobs.

Usage:
  python raft_build_data.py            # writes raft_data/train.jsonl
"""
import glob
import json
import os
import sys

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import nlp_manager as nm
from chunking import chunk_corpus
from retrieval import Retriever

DATA_DIR = "/home/jupyter/novice/nlp"
OUT_DIR = os.path.join(os.path.dirname(__file__), "raft_data")
HOLDOUT_N = 150  # first 150 questions reserved for eval, never trained on


def build_context(chunks, tokenizer):
    """Pack reranked chunks into a sources string within the token budget.

    Mirrors NLPManager._build_context so training context == inference context.
    """
    parts, used = [], 0
    for c in chunks:
        ids = tokenizer(c.text, add_special_tokens=False)["input_ids"]
        if used + len(ids) > nm.MAX_CONTEXT_TOKENS:
            ids = ids[:nm.MAX_CONTEXT_TOKENS - used]
            if ids:
                parts.append(tokenizer.decode(ids, skip_special_tokens=True))
            break
        parts.append(c.text)
        used += len(ids)
    return "\n\n".join(f"Source doc_{i}: {t}" for i, t in enumerate(parts))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # Use the host model cache (the container uses /app/models).
    nm.MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = {"torch_dtype": torch.float16} if device == "cuda" else {}

    print(">>> loading embedder + reranker (no vLLM)", flush=True)
    embedder = SentenceTransformer(
        nm.EMBEDDER_ID, cache_folder=nm.MODELS_DIR, local_files_only=True,
        device=device, model_kwargs=fp16)
    reranker = CrossEncoder(
        nm.RERANKER_ID, cache_folder=nm.MODELS_DIR, local_files_only=True,
        device=device, model_kwargs=fp16)
    llm_tok = AutoTokenizer.from_pretrained(
        nm._resolve_model_path(nm.PHI_MODEL_ID))

    doc_paths = sorted(glob.glob(f"{DATA_DIR}/documents/*.txt"))
    documents = [open(p).read() for p in doc_paths]
    chunks = chunk_corpus(documents, embedder.tokenizer)
    retriever = Retriever(embedder, reranker)
    retriever.index(chunks)
    print(f">>> indexed {len(chunks)} chunks from {len(documents)} docs",
          flush=True)

    rows = [json.loads(l) for l in open(f"{DATA_DIR}/nlp.jsonl") if l.strip()]
    train_rows = rows[HOLDOUT_N:]
    print(f">>> building {len(train_rows)} training examples "
          f"(holdout = first {HOLDOUT_N})", flush=True)

    questions = [r["question"] for r in train_rows]
    retrieved = retriever.retrieve_batch(questions)

    n_written = 0
    with open(os.path.join(OUT_DIR, "train.jsonl"), "w") as f:
        for r, chunk_list in zip(train_rows, retrieved):
            gold = (r.get("answer") or "").strip()
            if not gold or not chunk_list:
                continue  # no usable target or no context
            context = build_context(chunk_list, llm_tok)
            example = {
                "messages": [
                    {"role": "system", "content": nm.SYSTEM_PROMPT},
                    {"role": "user", "content":
                        f"SOURCES:\n{context}\n\nQUESTION: {r['question']}"},
                    {"role": "assistant", "content": f"FINAL ANSWER: {gold}"},
                ],
                "difficulty": r.get("difficulty", ""),
            }
            f.write(json.dumps(example) + "\n")
            n_written += 1

    print(f">>> wrote {n_written} examples to {OUT_DIR}/train.jsonl", flush=True)


if __name__ == "__main__":
    main()
