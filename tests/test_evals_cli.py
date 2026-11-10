import json
from pathlib import Path

import pytest

from centro_rag.cli import main as cli_main
from centro_rag.cli import parse_where
from centro_rag.evals import calibrate, load_cases, run, to_markdown
from centro_rag.evals import main as evals_main

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "synthetic_corpus"
CASES = ROOT / "evals" / "cases.jsonl"
THRESHOLDS = json.loads((ROOT / "evals" / "thresholds.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def report():
    return run(CORPUS, CASES, thresholds=THRESHOLDS)


def test_golden_run_has_no_failures(report):
    assert report.failed == 0, [c for c in report.cases if not c.passed] + report.threshold_failures
    assert report.embedder == "hashing-2048"


def test_metrics_are_split_by_group_and_strategy(report):
    assert set(report.group_metrics) == {"léxico", "parafraseo"}
    for by_strategy in report.group_metrics.values():
        assert set(by_strategy) == {"dense", "parent-section", "parent-document"}
    assert report.group_metrics["léxico"]["dense"]["n"] == 20
    assert report.group_metrics["parafraseo"]["dense"]["n"] == 10


def test_paraphrase_is_harder_than_lexical_for_the_baseline(report):
    """El baseline léxico no debe sacar lo mismo con sinónimos: si empata, el set no discrimina."""
    lex = report.group_metrics["léxico"]["dense"]
    par = report.group_metrics["parafraseo"]["dense"]
    assert par["mrr"] < lex["mrr"]


def test_parent_document_costs_more_context_than_chunks(report):
    m = report.metrics
    assert m["parent-document"]["avg_context_tokens"] > m["dense"]["avg_context_tokens"]
    assert report.baselines["tokens_corpus_completo"] > 0


def test_every_deterministic_case_ran(report):
    kinds = {c.type for c in report.cases}
    assert kinds == {"filter", "selfquery", "abstain"}
    assert len(report.cases) == len([c for c in load_cases(CASES) if c["type"] != "retrieval"])


def test_a_failing_case_is_detected(tmp_path):
    cases = tmp_path / "c.jsonl"
    cases.write_text(json.dumps({
        "id": "mal", "type": "filter", "query": "calibración del tomógrafo",
        "where": {"year": 2024}, "forbid_docs": ["mantenimiento-tomografo-2024"],
    }) + "\n", encoding="utf-8")
    rep = run(CORPUS, cases)
    assert rep.failed == 1 and "prohibidos" in rep.cases[0].detail


def test_unknown_case_type_fails(tmp_path):
    cases = tmp_path / "c.jsonl"
    cases.write_text(json.dumps({"id": "x", "type": "inventado"}) + "\n", encoding="utf-8")
    assert run(CORPUS, cases).failed == 1


def test_bad_json_line_reports_the_line_number(tmp_path):
    cases = tmp_path / "c.jsonl"
    cases.write_text('# comentario\n{"id": "ok", "type": "abstain", "query": "x"}\n{roto\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":3:"):
        load_cases(cases)


def test_unmet_threshold_makes_the_cli_fail(tmp_path):
    strict = tmp_path / "t.json"
    strict.write_text(json.dumps({"hashing-2048": {"parafraseo/dense": {"hit_at_k": 1.0}}}), encoding="utf-8")
    code = evals_main(["--corpus", str(CORPUS), "--cases", str(CASES), "--thresholds", str(strict)])
    assert code == 1


def test_cli_writes_reports_and_exits_zero(tmp_path):
    md, js = tmp_path / "r.md", tmp_path / "r.json"
    code = evals_main(["--corpus", str(CORPUS), "--cases", str(CASES),
                       "--thresholds", str(ROOT / "evals" / "thresholds.json"),
                       "--out-md", str(md), "--out-json", str(js)])
    assert code == 0
    text = md.read_text(encoding="utf-8")
    assert "parafraseo" in text and "Corpus sintético" in text and "FALLA" not in text
    assert json.loads(js.read_text(encoding="utf-8"))["embedder"] == "hashing-2048"


# ---------------------------------------------------------------- CLI de ingesta/búsqueda
def args(tmp_path, *extra, backend="json"):
    return ["--store", str(tmp_path / "store"), "--embedder", "hashing", "--backend", backend, *extra]


@pytest.mark.parametrize("backend", ["json", "chroma"])
def test_cli_ingest_is_idempotent_and_search_filters(tmp_path, capsys, backend):
    assert cli_main(["ingest", "--dir", str(CORPUS), *args(tmp_path, backend=backend)]) == 0
    assert "agregados=9" in capsys.readouterr().out
    assert cli_main(["ingest", "--dir", str(CORPUS), *args(tmp_path, backend=backend)]) == 0
    assert "sin_cambios=9" in capsys.readouterr().out
    code = cli_main(["search", "calibración del tomógrafo", "-k", "2", "--where", "year>=2024",
                    *args(tmp_path, backend=backend)])
    out = capsys.readouterr().out
    assert code == 0 and "mantenimiento-tomografo-2024.md" in out and "2019" not in out


def test_cli_prune_removes_documents_missing_from_the_source(tmp_path, capsys):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("Texto de a.", encoding="utf-8")
    (src / "b.txt").write_text("Texto de b.", encoding="utf-8")
    cli_main(["ingest", "--dir", str(src), *args(tmp_path)])
    (src / "b.txt").unlink()
    cli_main(["ingest", "--dir", str(src), "--prune", *args(tmp_path)])
    assert "borrados=1" in capsys.readouterr().out.splitlines()[-1]


def test_cli_document_level_returns_the_full_document(tmp_path, capsys):
    cli_main(["ingest", "--dir", str(CORPUS), *args(tmp_path)])
    capsys.readouterr()
    cli_main(["search", "licencia por estudio", "-k", "1", "--level", "document", *args(tmp_path)])
    assert "licencias-personal-2020.md" in capsys.readouterr().out


def test_cli_blocks_cloud_embedder_for_sensitive_data(tmp_path, capsys):
    code = cli_main(["ingest", "--dir", str(CORPUS), "--store", str(tmp_path / "s"),
                     "--embedder", "watsonx"])
    assert code == 2 and "sensitive" in capsys.readouterr().err


def test_cli_reports_bad_input_with_exit_2(tmp_path, capsys):
    assert cli_main(["ingest", "--dir", str(tmp_path / "no-existe"), *args(tmp_path)]) == 2
    assert cli_main(["search", "x", "--where", "año~2020", *args(tmp_path)]) == 2
    assert "Filtro inválido" in capsys.readouterr().err


def test_parse_where():
    assert parse_where(None) is None
    assert parse_where(["year>=2022", "institution=Hospital Modelo"]) == [
        ("year", "gte", 2022), ("institution", "eq", "Hospital Modelo")]
    with pytest.raises(ValueError):
        parse_where(["year>>2"])
    with pytest.raises(ValueError):
        parse_where(["year=abc"])


# ---------------------------------------------------------------- calibración de la abstención
class NeuralLikeEmbedder:
    """Imita a un modelo neuronal: lo irrelevante también tiene similitud apreciable con todo.

    Documentos -> [1, 0]. Consultas con respuesta -> coseno `hi`; sin respuesta -> coseno `lo`.
    Reproduce lo que pasó con el modelo multilingüe real, donde `min_score=0.15` no abstenía.
    """

    name, is_cloud = "neural-like", False

    def __init__(self, hi: float, lo: float):
        self.hi, self.lo = hi, lo

    @staticmethod
    def _vec(cos: float) -> list[float]:
        return [cos, (1 - cos * cos) ** 0.5]

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return self._vec(self.lo if "sin-respuesta" in text else self.hi)


@pytest.fixture
def two_cases(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text(
        json.dumps({"id": "in1", "type": "retrieval", "query": "consulta-con-respuesta",
                    "expect_docs": ["licencias-personal-2020"]}) + "\n"
        + json.dumps({"id": "out1", "type": "abstain", "query": "consulta-sin-respuesta"}) + "\n",
        encoding="utf-8",
    )
    return path


def test_a_fixed_threshold_from_the_lexical_baseline_fails_with_a_neural_model(two_cases):
    """Reproduce el fallo observado con el modelo local: 0.15 no abstiene si lo irrelevante da 0.3."""
    rep = run(CORPUS, two_cases, embedder=NeuralLikeEmbedder(hi=0.8, lo=0.3), min_score=0.15)
    abstain = next(c for c in rep.cases if c.id == "out1")
    assert not abstain.passed and "estado=answered" in abstain.detail
    assert rep.calibration["falsas_respuestas"] == ["out1"]


def test_calibration_finds_a_threshold_between_relevant_and_irrelevant(two_cases):
    rep = run(CORPUS, two_cases, embedder=NeuralLikeEmbedder(hi=0.8, lo=0.3))
    cal = rep.calibration
    assert cal["separable"] is True and cal["margen"] == 0.5
    assert cal["umbral_sugerido"] == 0.55 and cal["min_score_usado"] == 0.55
    assert cal["fuente"] == "calibrado en este mismo set"
    assert cal["falsas_abstenciones"] == [] and cal["falsas_respuestas"] == []
    assert next(c for c in rep.cases if c.id == "out1").passed


def test_overlapping_scores_leave_no_threshold_and_fail_loudly(two_cases):
    rep = run(CORPUS, two_cases, embedder=NeuralLikeEmbedder(hi=0.2, lo=0.4))
    assert rep.calibration["separable"] is False and rep.calibration["min_score_usado"] is None
    abstain = next(c for c in rep.cases if c.id == "out1")
    assert not abstain.passed and "sin umbral" in abstain.detail and rep.failed >= 1


def test_explicit_min_score_wins_over_calibration(two_cases):
    rep = run(CORPUS, two_cases, embedder=NeuralLikeEmbedder(hi=0.8, lo=0.3), min_score=0.9)
    assert rep.calibration["min_score_usado"] == 0.9
    assert rep.calibration["falsas_abstenciones"] == ["in1"]  # 0.8 < 0.9: rechazaría lo válido


def test_the_lexical_baseline_wrongly_abstains_on_many_paraphrased_questions(report):
    """Limitación medida: con 0.15, el baseline rechaza preguntas válidas redactadas con sinónimos."""
    wrong = report.calibration["falsas_abstenciones"]
    assert len(wrong) >= 3 and all(i.startswith("p") for i in wrong)


def test_the_lexical_baseline_cannot_separate_near_domain_questions(report):
    """Límite medido: con preguntas cercanas al dominio sin respuesta, ningún umbral separa."""
    cal = report.calibration
    assert cal["separable"] is False and cal["umbral_sugerido"] is None
    assert cal["falsas_respuestas"] and all("cercano" in i for i in cal["falsas_respuestas"])
    near = [c for c in report.cases if "cercano" in c.id]
    assert len(near) >= 10 and any(c.passed for c in near)  # no es todo-o-nada


def test_known_limits_are_shown_but_do_not_fail_the_run(report):
    limits = [c for c in report.cases if c.known_limit]
    assert limits and all(not c.passed for c in limits)
    assert report.failed == 0


def test_known_limit_only_applies_to_the_listed_embedder(tmp_path):
    case = {"id": "a", "type": "abstain", "query": "cuántos días de licencia por maternidad otorga el Ministerio de Salud Pública"}
    path = tmp_path / "c.jsonl"
    path.write_text(json.dumps({**case, "known_limit": ["hashing-2048"]}) + "\n", encoding="utf-8")
    assert run(CORPUS, path, min_score=0.15).failed == 0  # tolerado en hashing
    other = NeuralLikeEmbedder(hi=0.8, lo=0.9)  # cae en otro embedder: debe fallar de verdad
    assert run(CORPUS, path, embedder=other, min_score=0.15).failed == 1


def test_markdown_labels_known_limits(report):
    text = to_markdown(report)
    assert "LÍMITE CONOCIDO" in text and "FALLA" not in text


def test_calibrate_is_a_pure_function():
    ok = calibrate({"a": 0.6, "b": 0.8}, {"x": 0.2})
    assert ok["separable"] and ok["umbral_sugerido"] == 0.4 and ok["con_respuesta_mediana"] == 0.7
    bad = calibrate({"a": 0.1}, {"x": 0.3})
    assert bad["separable"] is False and bad["umbral_sugerido"] is None and bad["margen"] == -0.2
    empty = calibrate({}, {})
    assert empty["separable"] is None and empty["umbral_sugerido"] is None


def test_cli_min_score_flag_is_honored(tmp_path):
    base = ["--corpus", str(CORPUS), "--cases", str(CASES), "--thresholds", str(tmp_path / "ninguno.json")]
    assert evals_main([*base, "--min-score", "0.5"]) == 0
    assert evals_main([*base, "--min-score", "0.0"]) == 1  # 0.0 deja pasar consultas sin respuesta
