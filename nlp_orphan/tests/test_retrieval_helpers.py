"""Tests for the pure-logic retrieval helpers."""
from retrieval import bm25_tokenize, reciprocal_rank_fusion


def test_bm25_tokenize_lowercases_and_splits_on_nonalphanumeric():
    assert bm25_tokenize("Hello, World!") == ["hello", "world"]


def test_bm25_tokenize_decomposes_entity_codes_consistently():
    # An entity code in a question and a document must produce the same tokens.
    assert bm25_tokenize("EA-76-088") == ["ea", "76", "088"]


def test_bm25_tokenize_empty_string():
    assert bm25_tokenize("") == []


def test_rrf_merges_two_lists_favouring_items_ranked_high_in_both():
    dense = [10, 20, 30]
    sparse = [20, 40, 10]
    fused = reciprocal_rank_fusion([dense, sparse])
    # 20 is rank 1 in dense and rank 0 in sparse -> highest combined score.
    assert fused[0] == 20
    assert set(fused) == {10, 20, 30, 40}


def test_rrf_single_list_preserves_order():
    assert reciprocal_rank_fusion([[5, 6, 7]]) == [5, 6, 7]


def test_rrf_empty_input_returns_empty():
    assert reciprocal_rank_fusion([]) == []
