"""Hybrid dense + sparse retrieval with cross-encoder reranking."""
from __future__ import annotations

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
                 fuse_top=40, final_k=6):
        self.embedder = embedder
        self.reranker = reranker
        self.dense_top = dense_top
        self.sparse_top = sparse_top
        self.fuse_top = fuse_top
        self.final_k = final_k
        self.chunks: list[Chunk] = []
        self._chunk_emb = None
        self._bm25 = None
        self.last_top_score = None

    def index(self, chunks):
        """Build the dense embedding index and the BM25 index over chunks."""
        self.chunks = list(chunks)
        if not self.chunks:
            self._chunk_emb = None
            self._bm25 = None
            return
        texts = [c.text for c in self.chunks]
        self._chunk_emb = self.embedder.encode(
            texts, convert_to_tensor=True, normalize_embeddings=True,
            batch_size=64, show_progress_bar=False,
        )
        self._bm25 = BM25Okapi([bm25_tokenize(t) for t in texts])

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
        ranked = sorted(zip(fused, scores), key=lambda x: float(x[1]),
                        reverse=True)

        top = ranked[:self.final_k]
        self.last_top_score = float(top[0][1])
        return [self.chunks[i] for i, _ in top]
