import pytest

from centro_rag.llm import ScriptedLLM, StructuredOutputError
from centro_rag.retrieval import (
    DEFAULT_FIELDS,
    DenseRetriever,
    MultiQueryRetriever,
    ParentRetriever,
    PlanError,
    QueryPlan,
    SelfQueryRetriever,
    parse_variants,
    validate_plan,
)
from centro_rag.store import MemoryBackend

VOCAB = {
    "institution": ["Hospital Modelo", "Ministerio Modelo", "Instituto Modelo"],
    "document_type": ["protocolo", "normativa", "informe", "manual"],
}


def docs_of(hits):
    return [h.metadata["doc_id"] for h in hits]


# ---------------------------------------------------------------- padre
def test_parent_section_returns_the_whole_section(collection):
    child = DenseRetriever(collection).retrieve("recurso de revocación", 1)[0]
    parent = ParentRetriever(DenseRetriever(collection), collection, "section")
    hit = parent.retrieve("recurso de revocación", 1)[0]
    assert hit.id == child.metadata["section_id"]
    assert len(hit.text) >= len(child.text)
    assert hit.metadata["parent_level"] == "section"


def test_parent_document_returns_the_complete_document(collection, corpus_docs):
    parent = ParentRetriever(DenseRetriever(collection), collection, "document")
    hit = parent.retrieve("licencia por estudio", 1)[0]
    full = next(d for d in corpus_docs if d.doc_id == hit.id).parents[0].text
    assert hit.text == full
    assert "Licencia anual" in hit.text and "Solicitud" in hit.text  # varias secciones


def test_parents_are_deduplicated_and_count_matched_chunks(collection):
    parent = ParentRetriever(DenseRetriever(collection), collection, "document")
    hits = parent.retrieve("mantenimiento del tomógrafo calibración", 3)
    ids = [h.id for h in hits]
    assert len(ids) == len(set(ids))
    assert hits[0].metadata["matched_chunks"] >= 1


def test_missing_parent_degrades_to_the_chunk_and_says_so(collection):
    if isinstance(collection.backend, MemoryBackend):
        collection.backend._parents.clear()
    else:  # Chroma: se borran los padres sin tocar los chunks, simulando un índice inconsistente
        collection.backend._parents.delete(where={"doc_id": "licencias-personal-2020"})
    parent = ParentRetriever(DenseRetriever(collection), collection, "section")
    hit = parent.retrieve("licencia por estudio", 1)[0]
    assert hit.metadata.get("parent_missing") is True


def test_parent_respects_where_and_validates_overfetch(collection):
    parent = ParentRetriever(DenseRetriever(collection), collection, "section")
    hits = parent.retrieve("calibración", 3, {"year": 2019})
    assert hits and {h.metadata["year"] for h in hits} == {2019}
    with pytest.raises(ValueError):
        ParentRetriever(DenseRetriever(collection), collection, overfetch=0)


# ---------------------------------------------------------------- multi-query
def test_parse_variants_cleans_bullets_numbers_and_duplicates():
    text = "1. ¿Cuánto dura?\n- cuánto dura\n* Plazo de conservación\n\n3) \"Tiempo de guarda\"\n"
    assert parse_variants(text, 10) == ["¿Cuánto dura?", "Plazo de conservación", "Tiempo de guarda"]
    assert len(parse_variants(text, 2)) == 2


def test_multi_query_always_includes_the_original_and_fuses_with_rrf(collection):
    llm = ScriptedLLM(texts=["plazo de recurso contra la adjudicación\nconservación de historias"])
    mq = MultiQueryRetriever(DenseRetriever(collection), llm, n_variants=2)
    hits = mq.retrieve("cuántos días tiene el oferente para impugnar", 4)
    assert mq.last_queries[0] == "cuántos días tiene el oferente para impugnar"
    assert len(mq.last_queries) == 3
    assert "compras-normativa-2023" in docs_of(hits)
    assert all("rrf" in h.metadata for h in hits)
    assert len({h.id for h in hits}) == len(hits)


def test_multi_query_improves_a_query_the_baseline_misses(collection):
    plain = DenseRetriever(collection).retrieve("umbral económico para concurso abierto de ofertas", 3)
    assert "compras-normativa-2023" not in docs_of(plain)  # el baseline léxico falla acá
    llm = ScriptedLLM(texts=["monto a partir del cual corresponde licitación pública"])
    mq = MultiQueryRetriever(DenseRetriever(collection), llm, n_variants=1)
    assert "compras-normativa-2023" in docs_of(mq.retrieve("umbral económico para concurso abierto de ofertas", 3))


def test_multi_query_keeps_cosine_score_and_scope(collection):
    llm = ScriptedLLM(texts=["calibración del equipo"])
    mq = MultiQueryRetriever(DenseRetriever(collection), llm, n_variants=1)
    hits = mq.retrieve("calibración del tomógrafo", 5, {"year": 2019})
    assert hits and {h.metadata["year"] for h in hits} == {2019}
    assert all(0.0 <= h.score <= 1.0 for h in hits)


def test_multi_query_validation_and_identical_variants(collection):
    with pytest.raises(ValueError):
        MultiQueryRetriever(DenseRetriever(collection), ScriptedLLM(texts=["x"]), n_variants=0)
    mq = MultiQueryRetriever(DenseRetriever(collection), ScriptedLLM(texts=["Calibración del tomógrafo"]))
    assert mq.variants("calibración del tomógrafo") == ["calibración del tomógrafo"]


# ---------------------------------------------------------------- self-query
def plan(query="calibración", **filters):
    return {"query": query, "filters": [{"field": k, **v} for k, v in filters.items()]}


def test_self_query_applies_the_plan_filters(collection):
    llm = ScriptedLLM(structured=[plan(year={"op": "eq", "value": 2019})])
    sq = SelfQueryRetriever(DenseRetriever(collection), llm, vocabulary=VOCAB)
    hits = sq.retrieve("calibración del tomógrafo en 2019", 5)
    assert hits and {h.metadata["year"] for h in hits} == {2019}
    assert sq.last_plan.query == "calibración"


def test_self_query_retries_after_a_disallowed_field(collection):
    llm = ScriptedLLM(structured=[
        plan(autor={"op": "eq", "value": "x"}),
        plan(year={"op": "eq", "value": 2019}),
    ])
    sq = SelfQueryRetriever(DenseRetriever(collection), llm, vocabulary=VOCAB)
    hits = sq.retrieve("calibración", 3)
    assert {h.metadata["year"] for h in hits} == {2019}
    assert "Campo no permitido" in llm.prompts[1]


def test_self_query_gives_up_after_max_attempts(collection):
    llm = ScriptedLLM(structured=[plan(autor={"op": "eq", "value": "x"})])
    sq = SelfQueryRetriever(DenseRetriever(collection), llm, max_attempts=2)
    with pytest.raises(StructuredOutputError):
        sq.retrieve("calibración", 3)
    assert len(llm.prompts) == 2


def test_vocabulary_canonicalizes_case_and_rejects_unknown_values(collection):
    ok = QueryPlan.model_validate(plan(institution={"op": "eq", "value": "ministerio modelo"}))
    assert validate_plan(ok, DEFAULT_FIELDS, VOCAB) == [("institution", "eq", "Ministerio Modelo")]
    bad = QueryPlan.model_validate(plan(institution={"op": "eq", "value": "Hospital Inventado"}))
    with pytest.raises(PlanError, match="Válidos"):
        validate_plan(bad, DEFAULT_FIELDS, VOCAB)


@pytest.mark.parametrize(
    "filters",
    [
        {"year": {"op": "eq", "value": "2024"}},                 # texto en campo entero
        {"year": {"op": "eq", "value": True}},                   # bool no es entero
        {"institution": {"op": "gte", "value": "A"}},            # rango sobre texto
        {"year": {"op": "in", "value": 2024}},                   # 'in' sin lista
        {"year": {"op": "in", "value": []}},                     # 'in' vacío
        {"institution": {"op": "eq", "value": 5}},               # entero en campo de texto
    ],
)
def test_invalid_plans_are_rejected(filters):
    with pytest.raises(PlanError):
        validate_plan(QueryPlan.model_validate(plan(**filters)), DEFAULT_FIELDS)


def test_in_operator_is_validated_per_element():
    ok = QueryPlan.model_validate(plan(year={"op": "in", "value": [2019, 2020]}))
    assert validate_plan(ok, DEFAULT_FIELDS) == [("year", "in", [2019, 2020])]


def test_callers_scope_cannot_be_widened_by_the_llm(collection):
    llm = ScriptedLLM(structured=[plan(institution={"op": "eq", "value": "Ministerio Modelo"})])
    sq = SelfQueryRetriever(DenseRetriever(collection), llm, vocabulary=VOCAB)
    assert sq.retrieve("documentos", 5, {"institution": "Hospital Modelo"}) == []
    hits = SelfQueryRetriever(
        DenseRetriever(collection), ScriptedLLM(structured=[plan(year={"op": "gte", "value": 2022})]),
        vocabulary=VOCAB,
    ).retrieve("documentos", 10, {"institution": "Hospital Modelo"})
    assert hits and {h.metadata["institution"] for h in hits} == {"Hospital Modelo"}
    assert all(h.metadata["year"] >= 2022 for h in hits)
