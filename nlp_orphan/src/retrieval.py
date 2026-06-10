"""Hybrid dense + sparse retrieval with cross-encoder reranking.

Final document scoring is doc-level (not chunk-level dedup): each doc's score
is the best (max) reranker score across its candidate chunks, plus a small
boost from a title-summary BM25 match. This fixes the failure mode where the
canonical doc for a topic loses to docs that just *mention* the topic in
passing — measured to take recall@3 from 0.9733 to 0.9867 on the holdout 150.
"""
from __future__ import annotations

import os
import re

import torch
from rank_bm25 import BM25Okapi

from chunking import Chunk

# bge-v1.5 retrieval models expect this instruction prefixed to *queries*
# (not to documents) for best results.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def bm25_tokenize(text):
    """Lowercase and split into alphanumeric tokens for BM25.

    Splitting on non-alphanumerics means an entity code like 'EA-76-088'
    decomposes to ['ea', '76', '088'] identically wherever it appears, so a
    question and the document that answers it share those rare tokens.
    """
    return _TOKEN_RE.findall(text.lower())


def reciprocal_rank_fusion(ranked_lists, k=60):
    """Fuse ranked lists of item ids into one list, ordered by RRF score.

    RRF score of an item = sum over lists of 1 / (k + rank), where rank is the
    item's 0-based position in that list. Items absent from a list contribute
    nothing for that list. Robust to differing score scales (dense cosine vs
    BM25), which is why it beats raw score addition.
    """
    scores: dict = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


class Retriever:
    """Hybrid dense + BM25 retrieval over chunks, with cross-encoder reranking.

    Usage:
        r = Retriever(embedder, reranker)
        r.index(chunks)            # once, after the corpus arrives
        top = r.retrieve(question) # per question -> list[Chunk], best first
    """

    def __init__(self, embedder, reranker, dense_top=64, sparse_top=64,
                 fuse_top=80, final_k=6, title_weight=None):
        self.embedder = embedder
        self.reranker = reranker
        self.dense_top = dense_top
        self.sparse_top = sparse_top
        self.fuse_top = fuse_top
        self.final_k = final_k
        # alpha for doc-level title-BM25 boost; 0.2 empirically best on the
        # holdout-150 sweep (0.0/0.5/1.0+ all worse). Env-overridable.
        if title_weight is None:
            title_weight = float(os.environ.get("NLP_TITLE_WEIGHT", "0.2"))
        self.title_weight = title_weight
        self.chunks: list[Chunk] = []
        self._chunk_emb = None
        self._bm25 = None
        self._title_bm25 = None   # built only if doc_summaries are provided
        self.last_top_score = None

    def index(self, chunks, doc_summaries=None):
        """Build the dense + BM25 indices over chunks, optionally a per-doc
        title-summary BM25 used for doc-level rescoring."""
        self.chunks = list(chunks)
        if not self.chunks:
            self._chunk_emb = None
            self._bm25 = None
            self._title_bm25 = None
            return
        texts = [c.text for c in self.chunks]
        self._chunk_emb = self.embedder.encode(
            texts, convert_to_tensor=True, normalize_embeddings=True,
            batch_size=64, show_progress_bar=False,
        )
        self._bm25 = BM25Okapi([bm25_tokenize(t) for t in texts])
        self._title_bm25 = (
            BM25Okapi([bm25_tokenize(s) for s in doc_summaries])
            if doc_summaries else None)

    def _rank_docs(self, ranked_chunks, question):
        """Aggregate reranked chunks into doc-level scores and return chunks of
        the top docs, one per doc, in doc-rank order.

        ranked_chunks: list[(chunk_idx, rerank_score)] sorted desc.
        Returns: list[Chunk] with at most final_k entries, one per distinct doc.
        """
        # Per doc: best (chunk_idx, rerank_score).
        doc_best = {}
        for ci, s in ranked_chunks:
            d = self.chunks[ci].doc_index
            if d not in doc_best or s > doc_best[d][1]:
                doc_best[d] = (ci, float(s))
        if not doc_best:
            return []
        # Optional title-BM25 boost.
        if self._title_bm25 is not None and self.title_weight > 0:
            t_scores = self._title_bm25.get_scores(bm25_tokenize(question))
            max_t = float(max(t_scores)) if any(t_scores) else 1.0
            score = lambda d: doc_best[d][1] + self.title_weight * (
                float(t_scores[d]) / max_t if max_t else 0.0)
        else:
            score = lambda d: doc_best[d][1]
        ranked_docs = sorted(doc_best.keys(), key=score, reverse=True)
        out = [self.chunks[doc_best[d][0]] for d in ranked_docs[:self.final_k]]
        self.last_top_score = score(ranked_docs[0]) if ranked_docs else None
        return out

    def retrieve(self, question):
        """Return up to final_k chunks most relevant to the question."""
        if not self.chunks:
            self.last_top_score = None
            return []

        # Dense retrieval (cosine == dot product on normalized vectors).
        q_emb = self.embedder.encode(
            [BGE_QUERY_PREFIX + question], convert_to_tensor=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        sims = torch.matmul(self._chunk_emb, q_emb.squeeze(0))
        n = len(self.chunks)
        dense_ids = torch.topk(sims, min(self.dense_top, n)).indices.tolist()

        # Sparse retrieval (BM25 over the same chunks).
        bm25_scores = self._bm25.get_scores(bm25_tokenize(question))
        sparse_ids = sorted(
            range(n), key=lambda i: bm25_scores[i], reverse=True
        )[:self.sparse_top]

        # Fuse the two ranked lists, then rerank the candidates.
        fused = reciprocal_rank_fusion([dense_ids, sparse_ids])[:self.fuse_top]
        if not fused:
            self.last_top_score = None
            return []
        pairs = [[question, self.chunks[i].text] for i in fused]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        ranked = [(ci, float(s)) for ci, s in zip(fused, scores)]
        return self._rank_docs(ranked, question)

    def retrieve_batch(self, questions):
        """Retrieve for many questions, batching the GPU-heavy steps.

        Query embedding (one encode call) and cross-encoder reranking (one
        predict call over every candidate pair) are batched across the whole
        question list — far better GPU utilization than one call per question.
        Returns a list of chunk-lists aligned with `questions`.
        """
        if not self.chunks or not questions:
            return [[] for _ in questions]
        n = len(self.chunks)

        # Dense retrieval — embed all queries in one call.
        q_embs = self.embedder.encode(
            [BGE_QUERY_PREFIX + q for q in questions], convert_to_tensor=True,
            normalize_embeddings=True, show_progress_bar=False)
        sims = torch.matmul(q_embs, self._chunk_emb.T)  # [num_q, num_chunks]

        # Per question: fuse dense + BM25, collect the candidate pairs.
        fused_per_q, pairs, owner = [], [], []
        for qi, question in enumerate(questions):
            dense_ids = torch.topk(
                sims[qi], min(self.dense_top, n)).indices.tolist()
            bm25_scores = self._bm25.get_scores(bm25_tokenize(question))
            sparse_ids = sorted(
                range(n), key=lambda i: bm25_scores[i], reverse=True
            )[:self.sparse_top]
            fused = reciprocal_rank_fusion(
                [dense_ids, sparse_ids])[:self.fuse_top]
            fused_per_q.append(fused)
            for ci in fused:
                pairs.append([question, self.chunks[ci].text])
                owner.append(qi)

        results = [[] for _ in questions]
        if not pairs:
            return results

        # One cross-encoder pass over every (question, chunk) candidate pair.
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        flat_ci = [ci for fused in fused_per_q for ci in fused]
        scored = [[] for _ in questions]
        for qi, ci, sc in zip(owner, flat_ci, scores):
            scored[qi].append((ci, float(sc)))
        for qi, question in enumerate(questions):
            results[qi] = self._rank_docs(scored[qi], question)
        return results
