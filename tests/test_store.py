import pytest

from centro_rag.embeddings import CloudNotAllowedError, HashingEmbedder
from centro_rag.ingest import ChunkConfig, ingest_file
from centro_rag.store import (
    ChromaBackend,
    Collection,
    EmbedderMismatchError,
    MemoryBackend,
    matches,
    normalize_where,
)

LONG = "---\nyear: 2024\ninstitution: Hospital Modelo\n---\n" + "\n\n".join(
    f"Párrafo {i} sobre calibración y mantenimiento. " + "detalle " * 30 for i in range(8)
)
SHORT = "---\nyear: 2024\ninstitution: Hospital Modelo\n---\nSolo queda un párrafo corto."


def doc_from(tmp_path, text, name="d.md", **kw):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return ingest_file(path, config=ChunkConfig(300, 40), **kw)


# ---------------------------------------------------------------- upsert / borrado
def test_sync_reports_and_is_idempotent(collection, corpus_docs):
    again = collection.sync(corpus_docs)
    assert len(again.unchanged) == len(corpus_docs)
    assert not again.added and not again.updated
    assert collection.doc_ids() == sorted(d.doc_id for d in corpus_docs)


def test_update_replaces_content_and_leaves_no_orphan_chunks(embedder, make_backend, tmp_path):
    col = Collection("col", embedder, make_backend())
    big = doc_from(tmp_path, LONG)
    assert col.upsert_document(big) == "added"
    n_big = col.count()
    assert n_big > 3
    small = doc_from(tmp_path, SHORT)  # mismo doc_id, contenido más corto
    assert col.upsert_document(small) == "updated"
    assert col.count() == len(small.chunks) < n_big
    assert all(h.metadata["doc_id"] == "d" for h in col.search("párrafo", k=50))
    assert not any("Párrafo 7" in h.text for h in col.search("calibración", k=50))


def test_delete_document_removes_chunks_and_parents(collection):
    before = collection.count()
    removed = collection.delete_document("licencias-personal-2020")
    assert removed > 0 and collection.count() == before - removed
    assert "licencias-personal-2020" not in collection.doc_ids()
    assert collection.parents(["licencias-personal-2020", "licencias-personal-2020#s0"]) == {}
    assert collection.delete_document("no-existe") == 0


def test_sync_prune_deletes_documents_missing_from_source(collection, corpus_docs):
    rep = collection.sync(corpus_docs[:-1], prune=True)
    assert rep.deleted == [corpus_docs[-1].doc_id]
    assert corpus_docs[-1].doc_id not in collection.doc_ids()


def test_parents_are_retrievable(collection):
    hit = collection.search("historias clínicas quince años", k=1)[0]
    parents = collection.parents([hit.metadata["section_id"], hit.metadata["doc_id"]])
    assert set(parents) == {hit.metadata["section_id"], hit.metadata["doc_id"]}
    assert len(parents[hit.metadata["doc_id"]]) >= len(parents[hit.metadata["section_id"]])


# ---------------------------------------------------------------- filtros
def test_filters_all_operators(collection):
    years = lambda **w: {h.metadata["year"] for h in collection.search("protocolo", k=50, where=w)}  # noqa: E731
    assert years(year=2024) == {2024}
    assert years(year={"$gte": 2023}) == {2023, 2024}
    assert years(year={"$lte": 2020}) == {2019, 2020}
    assert years(year={"$gt": 2023}) == {2024}
    assert years(year={"$lt": 2020}) == {2019}
    assert years(year={"$in": [2019, 2021]}) == {2019, 2021}
    assert years(year={"$gte": 2020, "$lte": 2022}) == {2020, 2021, 2022}


def test_filter_on_missing_field_matches_nothing(collection):
    assert collection.search("tomógrafo", k=5, where={"autor": "x"}) == []


def test_filter_combination_is_an_and(collection):
    hits = collection.search(
        "protocolo", k=50, where={"institution": "Hospital Modelo", "year": {"$gte": 2024}}
    )
    assert hits and {h.metadata["doc_id"] for h in hits} <= {
        "mantenimiento-tomografo-2024", "triage-guardia-2024"
    }


def test_normalize_where_rejects_bad_input():
    with pytest.raises(ValueError):
        normalize_where({"year": {"$regex": "x"}})
    with pytest.raises(ValueError):
        normalize_where({"year": {"$in": 5}})
    assert normalize_where(None) == [] and normalize_where({}) == []


def test_matches_handles_incomparable_types():
    assert matches({"year": "dos"}, [("year", "gte", 2020)]) is False


def test_k_and_empty_index_edge_cases(embedder, make_backend, collection):
    assert collection.search("x", k=0) == []
    assert Collection("vacia", embedder, make_backend("vacia")).search("x", k=3) == []


# ---------------------------------------------------------------- aislamiento (bug original)
def test_two_collections_in_one_process_do_not_share_data(embedder, make_backend, tmp_path):
    a = Collection("a", embedder, make_backend("a"))
    b = Collection("b", embedder, make_backend("b"))
    a.upsert_document(doc_from(tmp_path, "Texto exclusivo del usuario uno.", "uno.txt"))
    b.upsert_document(doc_from(tmp_path, "Texto exclusivo del usuario dos.", "dos.txt"))
    seen_a = {h.metadata["doc_id"] for h in a.search("texto exclusivo", k=50)}
    seen_b = {h.metadata["doc_id"] for h in b.search("texto exclusivo", k=50)}
    assert seen_a == {"uno"} and seen_b == {"dos"}
    assert a.doc_ids() == ["uno"] and b.doc_ids() == ["dos"]


def test_deleting_in_one_collection_does_not_affect_another(embedder, make_backend, tmp_path):
    a = Collection("a", embedder, make_backend("a"))
    b = Collection("b", embedder, make_backend("b"))
    doc = doc_from(tmp_path, "Contenido compartido en ambas.", "x.txt")
    a.upsert_document(doc)
    b.upsert_document(doc)
    a.delete_document("x")
    assert a.count() == 0 and b.count() == len(doc.chunks)


# ---------------------------------------------------------------- persistencia
def test_data_survives_reopening(embedder, make_backend, tmp_path):
    first = Collection("col", embedder, make_backend("col"))
    first.upsert_document(doc_from(tmp_path, LONG))
    n = first.count()
    reopened = Collection("col", embedder, make_backend("col"))
    assert reopened.count() == n and reopened.doc_ids() == ["d"]
    assert reopened.search("calibración", k=1)
    assert reopened.upsert_document(doc_from(tmp_path, LONG)) == "unchanged"


# ---------------------------------------------------------------- guardas
def test_cannot_reopen_with_a_different_embedder(embedder, make_backend):
    Collection("col", embedder, make_backend("col"))
    with pytest.raises(EmbedderMismatchError, match="hashing"):
        Collection("col", HashingEmbedder(256), make_backend("col"))


def test_cannot_change_sensitivity_of_an_existing_collection(embedder, make_backend):
    Collection("col", embedder, make_backend("col"), sensitivity="sensitive")
    with pytest.raises(EmbedderMismatchError, match="sensitive"):
        Collection("col", embedder, make_backend("col"), sensitivity="public")


def test_cloud_embedder_is_refused_for_sensitive_collections():
    class FakeCloud:
        name, is_cloud = "fake-cloud", True

        def embed_documents(self, texts):
            return [[1.0]] * len(texts)

        def embed_query(self, text):
            return [1.0]

    with pytest.raises(CloudNotAllowedError):
        Collection("c", FakeCloud())
    assert Collection("c", FakeCloud(), sensitivity="public").count() == 0


def test_invalid_vectors_from_the_embedder_are_rejected(make_backend, tmp_path):
    class Broken:
        name, is_cloud = "broken", False

        def embed_documents(self, texts):
            return [[float("nan"), 1.0]] * len(texts)

        def embed_query(self, text):
            return [1.0, 0.0]

    col = Collection("c", Broken(), make_backend("c"))
    with pytest.raises(RuntimeError, match="inválidos"):
        col.upsert_document(doc_from(tmp_path, "algo de texto"))
    assert col.count() == 0


# ---------------------------------------------------------------- equivalencia de backends
def test_memory_and_chroma_return_the_same_ranking(embedder, corpus_docs, tmp_path):
    mem = Collection("m", embedder, MemoryBackend())
    chroma = Collection("c", embedder, ChromaBackend(tmp_path / "ch", "c"))
    mem.sync(corpus_docs)
    chroma.sync(corpus_docs)
    for query, where in [
        ("calibración del tomógrafo", None),
        ("plazos de licitación", {"year": {"$gte": 2022}}),
        ("residuos del laboratorio", {"institution": "Instituto Modelo"}),
    ]:
        a, b = mem.search(query, 5, where), chroma.search(query, 5, where)
        # Los puntajes coinciden posición por posición...
        assert all(abs(x.score - y.score) < 1e-4 for x, y in zip(a, b, strict=True))
        # ...y el orden coincide donde hay señal. Entre empates en ~0 el orden es arbitrario.
        assert [h.id for h in a if h.score > 1e-6] == [h.id for h in b if h.score > 1e-6]
