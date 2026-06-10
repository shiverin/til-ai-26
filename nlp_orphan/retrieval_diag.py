"""Diagnose retrieval misses: where does the gold doc actually rank?

For each known recall@3 miss, runs retrieval with a wide candidate pool
(fuse_top=80) and reports the rank of the gold doc in the post-rerank
candidate list (by parent-doc dedup). This tells us whether a cap bump
fixes the problem or deeper work is needed.
"""
import glob
import os
import sys

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import nlp_manager as nm
from chunking import chunk_corpus
from retrieval import Retriever

DATA_DIR = "/home/jupyter/novice/nlp"

# The 4 known recall@3 misses on the holdout-150
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
    chunks = chunk_corpus(documents, embedder.tokenizer)
    print(f">>> {len(chunks)} chunks, {len(documents)} docs", flush=True)

    # Wide pool: see far down the ranked list, not just top-6
    wide = Retriever(embedder, reranker, dense_top=64, sparse_top=64,
                     fuse_top=80, final_k=80)
    wide.index(chunks)

    # Also: dense-only and BM25-only ranks of the gold doc, to see whether
    # the gold doc is even in the fused candidate pool.
    for q, gold in MISSES:
        print("=" * 72)
        print(f"Q: {q}\nGOLD: {gold}")
        top_chunks = wide.retrieve(q)
        # rank by first appearance of gold's parent doc
        rank_in_rerank = None
        seen_docs = []
        for i, c in enumerate(top_chunks):
            did = doc_ids[c.doc_index]
            if did not in seen_docs:
                seen_docs.append(did)
            if did == gold and rank_in_rerank is None:
                rank_in_rerank = i + 1
        # at what doc-dedup rank does the gold appear?
        gold_doc_rank = (seen_docs.index(gold) + 1) if gold in seen_docs else None

        # Dense-only rank of gold
        q_emb = embedder.encode(
            ["Represent this sentence for searching relevant passages: " + q],
            convert_to_tensor=True, normalize_embeddings=True,
            show_progress_bar=False)
        sims = torch.matmul(wide._chunk_emb, q_emb.squeeze(0))
        dense_ranked = torch.argsort(sims, descending=True).tolist()
        gold_dense_chunk = next(
            (r for r, ci in enumerate(dense_ranked)
             if doc_ids[chunks[ci].doc_index] == gold), None)

        # BM25-only rank of gold
        from retrieval import bm25_tokenize
        bm25_scores = wide._bm25.get_scores(bm25_tokenize(q))
        bm25_ranked = sorted(range(len(chunks)),
                             key=lambda i: bm25_scores[i], reverse=True)
        gold_bm25_chunk = next(
            (r for r, ci in enumerate(bm25_ranked)
             if doc_ids[chunks[ci].doc_index] == gold), None)

        print(f"  rank in reranked CHUNKS (out of {len(top_chunks)}): "
              f"{rank_in_rerank}")
        print(f"  rank in reranked DOCS (dedup): {gold_doc_rank}")
        print(f"  rank in dense (chunk-level): {gold_dense_chunk}")
        print(f"  rank in BM25  (chunk-level): {gold_bm25_chunk}")
        if gold_doc_rank:
            print(f"  -> top-3 distinct docs: {seen_docs[:3]}")
            print(f"  -> top-5 distinct docs: {seen_docs[:5]}")
        print()


if __name__ == "__main__":
    main()
