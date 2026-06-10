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

    def __init__(self, embedder, reranker, dense_top=50, sparse_top=50,
                 fuse_top=50, final_k=6, title_weight=None,
                 title_dense_weight=None, body_bm25_weight=None,
                 use_reranker=None):
        self.embedder = embedder
        self.reranker = reranker
        # fuse_top is the dominant runtime knob: reranker time scales linearly
        # with the candidate count. Sweep on holdout-150 showed fuse_top=50
        # holds recall@3=0.9933 at 2.9x the speed of fuse_top=80 (0.75 s/q
        # vs 2.19 s/q in the in-process eval). 40 regresses (drops a question).
        self.dense_top = int(os.environ.get("NLP_DENSE_TOP", str(dense_top)))
        self.sparse_top = int(os.environ.get("NLP_SPARSE_TOP", str(sparse_top)))
        self.fuse_top = int(os.environ.get("NLP_FUSE_TOP", str(fuse_top)))
        self.final_k = final_k
        # When disabled, the cross-encoder reranker pass is skipped entirely
        # (the dominant per-query cost). Chunks are scored by their position
        # in the RRF-fused list instead; doc-level boosts still apply.
        if use_reranker is None:
            use_reranker = os.environ.get("NLP_USE_RERANKER", "1") == "1"
        self.use_reranker = use_reranker
        # Doc-level boost weights. Title-BM25 catches verbatim entity hits in
        # the doc summary; title-dense (cosine vs query embedding) catches
        # semantic matches BM25 misses. Both env-overridable for tuning.
        if title_weight is None:
            title_weight = float(os.environ.get("NLP_TITLE_WEIGHT", "0.2"))
        if title_dense_weight is None:
            # 0.5 empirically best on the holdout-150 sweep (the BM25-only
            # title boost capped at recall@3 0.9867; adding the dense signal
            # at weight 0.5 took it to 0.9933).
            title_dense_weight = float(
                os.environ.get("NLP_TITLE_DENSE_WEIGHT", "0.5"))
        if body_bm25_weight is None:
            # Doc-level body-BM25 signal (max chunk-BM25 score per doc) was
            # trialled but did not move recall@3 at any weight (0.1..2.0): the
            # gap from chunk-level rerank dominated. Default off (no compute);
            # env-overridable for future experimentation.
            body_bm25_weight = float(
                os.environ.get("NLP_BODY_BM25_WEIGHT", "0.0"))
        self.title_weight = title_weight
        self.title_dense_weight = title_dense_weight
        self.body_bm25_weight = body_bm25_weight
        self.chunks: list[Chunk] = []
        self._chunk_emb = None
        self._bm25 = None
        self._title_bm25 = None      # BM25 over doc summaries
        self._summary_emb = None     # dense embeddings of doc summaries
        self.last_top_score = None

    def index(self, chunks, doc_summaries=None):
        """Build the dense + BM25 indices over chunks, optionally per-doc
        title-summary indices (BM25 + dense) used for doc-level rescoring."""
        self.chunks = list(chunks)
        if not self.chunks:
            self._chunk_emb = None
            self._bm25 = None
            self._title_bm25 = None
            self._summary_emb = None
            return
        texts = [c.text for c in self.chunks]
        self._chunk_emb = self.embedder.encode(
            texts, convert_to_tensor=True, normalize_embeddings=True,
            batch_size=64, show_progress_bar=False,
        )
        self._bm25 = BM25Okapi([bm25_tokenize(t) for t in texts])
        if doc_summaries:
            self._title_bm25 = BM25Okapi(
                [bm25_tokenize(s) for s in doc_summaries])
            # Dense embeddings of doc summaries (one vector per doc), for an
            # additional semantic doc-level signal.
            self._summary_emb = self.embedder.encode(
                doc_summaries, convert_to_tensor=True,
                normalize_embeddings=True, batch_size=32,
                show_progress_bar=False,
            )
        else:
            self._title_bm25 = None
            self._summary_emb = None

    def _rank_docs(self, ranked_chunks, question, q_emb=None):
        """Aggregate reranked chunks into doc-level scores and return chunks of
        the top docs, one per doc, in doc-rank order.

        ranked_chunks: list[(chunk_idx, rerank_score)] sorted desc.
        q_emb: pre-computed query embedding (1, D) — if None and the dense
            title signal is needed, it's encoded on the fly.
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

        # Title-BM25 score per doc, normalized to [0, 1].
        t_scores = None
        if self._title_bm25 is not None and self.title_weight > 0:
            raw = self._title_bm25.get_scores(bm25_tokenize(question))
            mx = float(max(raw)) if any(raw) else 0.0
            t_scores = (raw / mx) if mx else raw  # ndarray division ok
        # Dense doc-summary cosine, normalized to [0, 1].
        d_scores = None
        if (self._summary_emb is not None
                and self.title_dense_weight > 0):
            if q_emb is None:
                q_emb = self.embedder.encode(
                    [BGE_QUERY_PREFIX + question], convert_to_tensor=True,
                    normalize_embeddings=True, show_progress_bar=False)
            sims = torch.matmul(self._summary_emb, q_emb.squeeze(0))
            # Normalize to [0, 1] by max (cosine is in [-1, 1] but with
            # normalized vectors of similar topics, max is the natural scale).
            mx = float(sims.max()) if sims.numel() else 0.0
            d_scores = (sims / mx).tolist() if mx else sims.tolist()

        # Body-BM25 per doc: max chunk-BM25 score over candidate chunks of the
        # doc. Captures content-level discrimination the title summary misses.
        b_scores = None
        if self.body_bm25_weight > 0 and self._bm25 is not None:
            raw = self._bm25.get_scores(bm25_tokenize(question))
            doc_body_max = {}
            for ci, sc in enumerate(raw):
                d = self.chunks[ci].doc_index
                if d in doc_best:  # only consider docs already in the pool
                    cur = doc_body_max.get(d, 0.0)
                    if float(sc) > cur:
                        doc_body_max[d] = float(sc)
            if doc_body_max:
                mx = max(doc_body_max.values())
                if mx > 0:
                    b_scores = {d: s / mx for d, s in doc_body_max.items()}

        def score(d):
            s = doc_best[d][1]
            if t_scores is not None:
                s += self.title_weight * float(t_scores[d])
            if d_scores is not None:
                s += self.title_dense_weight * float(d_scores[d])
            if b_scores is not None:
                s += self.body_bm25_weight * b_scores.get(d, 0.0)
            return s

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
        if self.use_reranker:
            pairs = [[question, self.chunks[i].text] for i in fused]
            scores = self.reranker.predict(pairs, show_progress_bar=False)
            ranked = [(ci, float(s)) for ci, s in zip(fused, scores)]
        else:
            # Skip reranker: chunk-level score = 1/(rank+1) from RRF order.
            ranked = [(ci, 1.0 / (i + 1)) for i, ci in enumerate(fused)]
        return self._rank_docs(ranked, question, q_emb=q_emb)

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
            if self.use_reranker:
                for ci in fused:
                    pairs.append([question, self.chunks[ci].text])
                    owner.append(qi)

        results = [[] for _ in questions]
        if self.use_reranker and pairs:
            # One cross-encoder pass over every (question, chunk) pair.
            scores = self.reranker.predict(pairs, show_progress_bar=False)
            flat_ci = [ci for fused in fused_per_q for ci in fused]
            scored = [[] for _ in questions]
            for qi, ci, sc in zip(owner, flat_ci, scores):
                scored[qi].append((ci, float(sc)))
        else:
            # No reranker: chunk score = 1/(rank+1) from RRF order.
            scored = [
                [(ci, 1.0 / (i + 1)) for i, ci in enumerate(fused)]
                for fused in fused_per_q
            ]
        for qi, question in enumerate(questions):
            results[qi] = self._rank_docs(
                scored[qi], question, q_emb=q_embs[qi:qi + 1])
        return results
