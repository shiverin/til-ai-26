"""Diagnostic 2: test whether a document-level title-match signal would fix
the 4 recall@3 misses without overfitting.

For each miss, computes:
  - rank of gold doc by title-BM25 alone (titles = H1 + all headers)
  - rank of gold doc by combined score (reranker_max_chunk + alpha*title_norm)
    across several alpha values
"""
import glob
import os
import re
import sys

import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import nlp_manager as nm
from chunking import chunk_corpus
from retrieval import Retriever, bm25_tokenize

DATA_DIR = "/home/jupyter/novice/nlp"
HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

MISSES = [
    ("Which four organizations signed the Cordial Entente in September 2071?",
     "DOC-0013"),
    ("What dual executive roles does Cary Mullin hold at Genesis Labs?",
     "DOC-0264"),
    ("In Park Soo-Hyun's first 90-day strategic review, are Renhwa's elevated "
     "content production costs likely to decrease or persist in the near "
     "term? Explain.",
     "DOC-0019"),
    ("Given the memo's assessment of inter-megacorp trust after the Wampa "
     "action, what type of provision did Phyrexis Group propose to "
     "compensate for that condition?",
     "DOC-0010"),
]


def doc_title_string(doc_text):
    """Title + all headers from a markdown doc."""
    return " | ".join(m.group(2).strip() for m in HEADER_RE.finditer(doc_text))


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

    # NEW: per-doc title strings (all headers concatenated)
    doc_titles = [doc_title_string(d) for d in documents]
    title_bm25 = BM25Okapi([bm25_tokenize(t) for t in doc_titles])

    chunks = chunk_corpus(documents, embedder.tokenizer)
    retriever = Retriever(embedder, reranker, dense_top=64, sparse_top=64,
                          fuse_top=80, final_k=80)
    retriever.index(chunks)
    print(f">>> indexed {len(chunks)} chunks", flush=True)

    alphas = [0.0, 0.3, 0.5, 1.0, 2.0, 5.0]

    for q, gold in MISSES:
        print("=" * 72)
        print(f"Q: {q[:80]}\nGOLD: {gold}")

        # Title-BM25 alone, doc-level ranking
        t_scores = title_bm25.get_scores(bm25_tokenize(q))
        title_ranked = sorted(range(len(doc_titles)),
                              key=lambda d: t_scores[d], reverse=True)
        title_only_rank = title_ranked.index(doc_ids.index(gold)) + 1
        print(f"  title-only doc rank: {title_only_rank}  "
              f"(score={t_scores[doc_ids.index(gold)]:.3f}, "
              f"top score={t_scores[title_ranked[0]]:.3f})")

        # Retrieve wide pool, get reranker scores per chunk
        top_chunks = retriever.retrieve(q)  # returns up to final_k=80 chunks
        # We need (chunk_idx, rerank_score). Retriever doesn't expose scores per
        # chunk publicly except last_top_score. Re-run the fusion+rerank here.
        q_emb = embedder.encode(
            ["Represent this sentence for searching relevant passages: " + q],
            convert_to_tensor=True, normalize_embeddings=True,
            show_progress_bar=False)
        sims = torch.matmul(retriever._chunk_emb, q_emb.squeeze(0))
        dense_ids = torch.topk(sims, 64).indices.tolist()
        sparse_scores = retriever._bm25.get_scores(bm25_tokenize(q))
        sparse_ids = sorted(range(len(chunks)),
                            key=lambda i: sparse_scores[i], reverse=True)[:64]
        from retrieval import reciprocal_rank_fusion
        fused = reciprocal_rank_fusion([dense_ids, sparse_ids])[:80]
        pairs = [[q, chunks[i].text] for i in fused]
        scores = reranker.predict(pairs, show_progress_bar=False)

        # Per-doc max rerank score (over chunks in the candidate pool)
        doc_max = {}
        for ci, s in zip(fused, scores):
            d = chunks[ci].doc_index
            doc_max[d] = max(doc_max.get(d, -1e9), float(s))

        # Normalize title scores 0..1
        max_t = max(t_scores) if any(t_scores) else 1.0

        # Try several alphas, report rank of gold doc
        print(f"  rerank_max_chunk + alpha * (title_score/max):")
        for alpha in alphas:
            ranked_docs = sorted(
                doc_max.keys(),
                key=lambda d: doc_max[d] + alpha * (t_scores[d] / max_t),
                reverse=True)
            gold_idx = doc_ids.index(gold)
            if gold_idx in ranked_docs:
                gold_rank = ranked_docs.index(gold_idx) + 1
            else:
                gold_rank = f">{len(ranked_docs)} (not in candidates)"
            top3_ids = [doc_ids[d] for d in ranked_docs[:3]]
            print(f"    alpha={alpha:>4}: gold rank = {gold_rank}  "
                  f"top3={top3_ids}")
        print()


if __name__ == "__main__":
    main()
