"""Hybrid dense + sparse retrieval with cross-encoder reranking."""
from __future__ import annotations

import re

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
