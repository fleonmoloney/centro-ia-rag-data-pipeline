import sys
import types

import numpy as np
import pytest

from centro_rag.embeddings import (
    CloudNotAllowedError,
    HashingEmbedder,
    WatsonxEmbedder,
    build_embedder,
)


def cos(a, b):
    return float(np.dot(a, b))


def test_deterministic_and_normalized():
    e = HashingEmbedder()
    v1, v2 = e.embed_query("calibración del tomógrafo"), e.embed_query("calibración del tomógrafo")
    assert v1 == v2
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-9
    assert len(v1) == 2048


def test_ignores_case_accents_and_stopwords():
    e = HashingEmbedder()
    assert cos(e.embed_query("Calibración del Tomógrafo"), e.embed_query("calibracion tomografo")) > 0.99


def test_related_text_scores_higher_than_unrelated():
    e = HashingEmbedder()
    q = e.embed_query("plazo de conservación de historias clínicas")
    near = e.embed_documents(["Las historias clínicas se conservan por quince años."])[0]
    far = e.embed_documents(["El tomógrafo se calibra cada seis meses."])[0]
    assert cos(q, near) > cos(q, far) + 0.2


def test_empty_text_gives_zero_vector_not_nan():
    v = HashingEmbedder().embed_query("")
    assert all(x == 0.0 for x in v)


def test_dim_validation_and_name():
    with pytest.raises(ValueError):
        HashingEmbedder(8)
    assert HashingEmbedder(256).name == "hashing-256"


def test_cloud_embedder_is_blocked_for_sensitive_data():
    with pytest.raises(CloudNotAllowedError):
        build_embedder("watsonx", sensitivity="sensitive")


def test_unknown_embedder_name():
    with pytest.raises(ValueError):
        build_embedder("nada")


def test_watsonx_requires_credentials(monkeypatch):
    monkeypatch.delenv("WATSONX_API_KEY", raising=False)
    monkeypatch.delenv("WATSONX_PROJECT_ID", raising=False)
    with pytest.raises(RuntimeError, match="WATSONX_API_KEY"):
        WatsonxEmbedder()


def test_watsonx_wiring_with_a_fake_sdk(monkeypatch):
    """Verifica cómo se arma la llamada al SDK sin red ni cuenta (SDK reemplazado por un fake)."""
    calls = {}

    class FakeCredentials:
        def __init__(self, **kw):
            calls["credentials"] = kw

    class FakeEmbeddings:
        def __init__(self, **kw):
            calls["embeddings"] = kw

        def embed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

        def embed_query(self, text):
            return [0.0, 1.0]

    pkg = types.ModuleType("ibm_watsonx_ai")
    pkg.Credentials = FakeCredentials
    sub = types.ModuleType("ibm_watsonx_ai.foundation_models")
    sub.Embeddings = FakeEmbeddings
    monkeypatch.setitem(sys.modules, "ibm_watsonx_ai", pkg)
    monkeypatch.setitem(sys.modules, "ibm_watsonx_ai.foundation_models", sub)

    emb = build_embedder_for_test(monkeypatch)
    assert emb.embed_documents(["a", "b"]) == [[1.0, 0.0], [1.0, 0.0]]
    assert emb.embed_query("q") == [0.0, 1.0]
    assert emb.name == "watsonx:ibm/granite-embedding-278m-multilingual"
    assert calls["credentials"]["api_key"] == "k"
    assert calls["embeddings"]["project_id"] == "p"
    assert calls["embeddings"]["model_id"] == "ibm/granite-embedding-278m-multilingual"


def build_embedder_for_test(monkeypatch):
    monkeypatch.setenv("WATSONX_API_KEY", "k")
    monkeypatch.setenv("WATSONX_PROJECT_ID", "p")
    return build_embedder("watsonx", sensitivity="public")
