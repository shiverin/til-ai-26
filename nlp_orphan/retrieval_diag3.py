"""Diagnostic 3: full 150-question recall@3 sweep with doc-level
aggregation + title-summary boost.

For each candidate alpha, compute recall@3 across all 150 holdout questions
using:
    doc_score(d) = max_chunk_rerank_score(d) + alpha * (title_bm25(d) / max_title)
where the "title" for BM25 is the first 600 chars of each doc (captures
markdown headers AND `**Subject:**` / `**RE:**` / `**FROM:**` metadata).

Picks the smallest alpha that fixes the misses without breaking other questions.
"""
import collections
import glob
import json
import os
import sys

import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import nlp_manager as nm
from chunking import chunk_corpus
from retrieval import Retriever, bm25_tokenize, reciprocal_rank_fusion

DATA_DIR = "/home/jupyter/novice/nlp"
SUMMARY_CHARS = 600  # capture title + classification + metadata + opening


def main():
    nm.MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = {"torch_dtype": torch.float16} if device == "cuda" else {}

    print(">>> loading embedder + reranker", flush=True)
    embedder = SentenceTransformer(
        nm.EMBEDDER_ID, cache_folder=nm.MODELS_DIR, local_files_only=True,
        device=device, model_kwargs=fp16)
    reranker = CrossEncoder(
        nm.RERANKER_ID, cache_folder=nm.MODELS_DIR, local_files_only=True,
        device=device, model_kwargs=fp16)

    doc_paths = sorted(glob.glob(f"{DATA_DIR}/documents/*.txt"))
    doc_ids = [os.path.basename(p)[:-4] for p in doc_paths]
    documents = [open(p).read() for p in doc_paths]
    doc_summaries = [d[:SUMMARY_CHARS] for d in documents]
    title_bm25 = BM25Okapi([bm25_tokenize(s) for s in doc_summaries])

    chunks = chunk_corpus(documents, embedder.tokenizer)
    retriever = Retriever(embedder, reranker, dense_top=64, sparse_top=64,
                          fuse_top=80, final_k=80)
    retriever.index(chunks)
    print(f">>> indexed {len(chunks)} chunks", flush=True)

    rows = [json.loads(l) for l in open(f"{DATA_DIR}/nlp.jsonl") if l.strip()][:150]
    print(f">>> evaluating {len(rows)} questions", flush=True)

    # Batch retrieve (faster than one-by-one)
    questions = [r["question"] for r in rows]
    # Need per-chunk rerank scores -> re-run pipeline manually, batched.
    q_embs = embedder.encode(
        ["Represent this sentence for searching relevant passages: " + q
         for q in questions],
        convert_to_tensor=True, normalize_embeddings=True,
        show_progress_bar=False, batch_size=32)
    sims_all = torch.matmul(q_embs, retriever._chunk_emb.T)  # [Q, C]

    alphas = [0.0, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    hits_at_3 = {a: 0 for a in alphas}
    miss_qs = {a: [] for a in alphas}

    for qi, (q, row) in enumerate(zip(questions, rows)):
        gold = set(row["source_docs"])
        # Dense top
        dense_ids = torch.topk(sims_all[qi], 64).indices.tolist()
        # Sparse top
        sparse_scores = retriever._bm25.get_scores(bm25_tokenize(q))
        sparse_ids = sorted(range(len(chunks)),
                            key=lambda i: sparse_scores[i], reverse=True)[:64]
        # Fuse
        fused = reciprocal_rank_fusion([dense_ids, sparse_ids])[:80]
        # Rerank
        pairs = [[q, chunks[i].text] for i in fused]
        scores = reranker.predict(pairs, show_progress_bar=False)
        # Per-doc max rerank
        doc_max = {}
        for ci, s in zip(fused, scores):
            d = chunks[ci].doc_index
            doc_max[d] = max(doc_max.get(d, -1e9), float(s))
        # Title bm25
        t_scores = title_bm25.get_scores(bm25_tokenize(q))
        max_t = max(t_scores) if any(t_scores) else 1.0

        for alpha in alphas:
            ranked_docs = sorted(
                doc_max.keys(),
                key=lambda d: doc_max[d] + alpha * (t_scores[d] / max_t),
                reverse=True)
            top3 = [doc_ids[d] for d in ranked_docs[:3]]
            if gold & set(top3):
                hits_at_3[alpha] += 1
            else:
                miss_qs[alpha].append((qi, q[:60], list(gold), top3))

        if (qi + 1) % 30 == 0:
            print(f"  {qi+1}/{len(rows)}", flush=True)

    print("\n=== RECALL@3 BY ALPHA ===")
    for a in alphas:
        rec = hits_at_3[a] / len(rows)
        print(f"  alpha={a:>4}: recall@3 = {rec:.4f} "
              f"({hits_at_3[a]}/{len(rows)})  | misses={len(miss_qs[a])}")

    # Show misses only at best (highest) alpha
    best_alpha = max(alphas, key=lambda a: hits_at_3[a])
    print(f"\n=== MISSES at best alpha={best_alpha} ({len(miss_qs[best_alpha])}) ===")
    for qi, q, gold, top3 in miss_qs[best_alpha]:
        print(f"  qi={qi}  gold={gold}  top3={top3}")
        print(f"    Q: {q}")


if __name__ == "__main__":
    main()
