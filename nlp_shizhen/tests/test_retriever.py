"""Integration test for the Retriever (loads real embedder + reranker)."""
import pytest
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from chunking import Chunk
from retrieval import Retriever

MODELS = "/home/jupyter/til-ai-26/nlp_shizhen/models"


@pytest.fixture(scope="module")
def retriever():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(
        "BAAI/bge-large-en-v1.5", cache_folder=MODELS,
        local_files_only=True, device=device,
    )
    reranker = CrossEncoder(
        "BAAI/bge-reranker-v2-m3", cache_folder=MODELS,
        local_files_only=True, device=device,
    )
    return Retriever(embedder, reranker, final_k=2)


def test_retrieves_the_chunk_that_answers_the_question(retriever):
    chunks = [
        Chunk(text="The capital of France is Paris.", doc_index=0),
        Chunk(text="Photosynthesis converts sunlight into chemical energy.", doc_index=1),
        Chunk(text="The Halcyon penalty was 4.5 million Credits.", doc_index=2),
    ]
    retriever.index(chunks)
    got = retriever.retrieve("What penalty did Halcyon receive?")
    assert any(c.doc_index == 2 for c in got)
    assert retriever.last_top_score is not None


def test_empty_index_returns_empty(retriever):
    retriever.index([])
    assert retriever.retrieve("anything") == []
