import os
from pathlib import Path

import pytest

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from centro_rag.embeddings import HashingEmbedder
from centro_rag.ingest import ingest_directory
from centro_rag.store import ChromaBackend, Collection, MemoryBackend

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "synthetic_corpus"


@pytest.fixture(scope="session")
def corpus_docs():
    docs, skipped = ingest_directory(CORPUS)
    assert not skipped
    return docs


@pytest.fixture
def embedder():
    return HashingEmbedder()


@pytest.fixture(params=["memory", "chroma"])
def make_backend(request, tmp_path):
    """Fábrica de backends: cada llamada con el mismo `name` y `tmp_path` reabre lo persistido."""

    def factory(name: str = "col"):
        if request.param == "memory":
            return MemoryBackend(tmp_path / f"{name}.json")
        return ChromaBackend(tmp_path / "chroma", name)

    factory.kind = request.param
    return factory


@pytest.fixture
def collection(embedder, make_backend, corpus_docs):
    col = Collection("col", embedder, make_backend("col"))
    col.sync(corpus_docs)
    return col
