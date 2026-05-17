"""Tests for the header-aware Markdown chunker."""
from chunking import Chunk, chunk_document, chunk_corpus


class FakeTokenizer:
    """Whitespace tokenizer — ids are the words themselves."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


def test_returns_chunk_objects_with_doc_index():
    doc = "# Title\n\nSome body text here."
    chunks = chunk_document(doc, doc_index=7, tokenizer=FakeTokenizer())
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.doc_index == 7 for c in chunks)


def test_prepends_title_and_section_header():
    doc = "# Big Title\n\n## Section One\n\nalpha beta gamma"
    chunks = chunk_document(doc, doc_index=0, tokenizer=FakeTokenizer())
    assert "Big Title" in chunks[0].text
    assert "Section One" in chunks[0].text
    assert "alpha beta gamma" in chunks[0].text


def test_long_section_splits_into_multiple_overlapping_chunks():
    body = " ".join(f"w{i}" for i in range(100))
    doc = f"# T\n\n## S\n\n{body}"
    chunks = chunk_document(
        doc, doc_index=0, tokenizer=FakeTokenizer(),
        target_tokens=40, overlap_tokens=10,
    )
    assert len(chunks) >= 3
    # consecutive chunks overlap: last words of chunk[0] reappear in chunk[1]
    assert "w39" in chunks[0].text
    assert "w30" in chunks[1].text


def test_empty_or_headerless_doc_yields_one_fallback_chunk():
    chunks = chunk_document("plain text no headers", 3, FakeTokenizer())
    assert len(chunks) == 1
    assert chunks[0].doc_index == 3
    assert "plain text" in chunks[0].text


def test_chunk_corpus_indexes_each_document():
    docs = ["# A\n\nfirst doc body", "# B\n\nsecond doc body"]
    chunks = chunk_corpus(docs, FakeTokenizer())
    assert {c.doc_index for c in chunks} == {0, 1}
