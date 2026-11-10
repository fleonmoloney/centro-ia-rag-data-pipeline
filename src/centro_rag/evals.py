"""Evals de recuperación sobre un corpus con casos dorados.

Mide, por estrategia (chunks, padre-sección, padre-documento):
* hit@k   : en cuántas consultas aparece algún documento esperado entre los k primeros.
* MRR     : posición del primer documento esperado.
* tokens  : contexto devuelto, comparado con el documento completo y con todo el corpus.

Y verifica con casos deterministas: filtros por metadata, planes de self-query (con LLM de
guion: prueba la validación y el alcance, no la calidad de ningún modelo) y abstención.

Los tokens son una estimación (~4 caracteres/token). Las métricas dependen del corpus y del
embedder: con el corpus sintético y `hashing` miden un baseline léxico, no calidad semántica.

Uso: python -m centro_rag.evals --corpus data/synthetic_corpus --cases evals/cases.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median

from centro_rag.embeddings import Embedder, Sensitivity, build_embedder
from centro_rag.ingest import IngestedDoc, ingest_directory
from centro_rag.llm import ScriptedLLM
from centro_rag.qa import answer
from centro_rag.retrieval import DenseRetriever, ParentRetriever, Retriever, SelfQueryRetriever
from centro_rag.store import Collection

STRATEGIES = ("dense", "parent-section", "parent-document")


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class CaseResult:
    id: str
    type: str
    passed: bool
    detail: str = ""
    known_limit: bool = False  # falla esperada y documentada para este embedder (no cuenta)


@dataclass
class Report:
    embedder: str
    k: int
    cases: list[CaseResult] = field(default_factory=list)
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    group_metrics: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    baselines: dict[str, float] = field(default_factory=dict)
    threshold_failures: list[str] = field(default_factory=list)
    misses: dict[str, list[str]] = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)

    @property
    def failed(self) -> int:
        return (
            sum(1 for c in self.cases if not c.passed and not c.known_limit)
            + len(self.threshold_failures)
        )


def load_cases(path: str | Path) -> list[dict]:
    cases = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if line and not line.startswith("#"):
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{n}: JSON inválido ({exc})") from exc
    return cases


def vocabulary_from(docs: list[IngestedDoc]) -> dict[str, list[str]]:
    vocab: dict[str, set[str]] = {"institution": set(), "document_type": set()}
    for doc in docs:
        for key in vocab:
            if isinstance(doc.metadata.get(key), str):
                vocab[key].add(doc.metadata[key])
    return {k: sorted(v) for k, v in vocab.items()}


def _doc_ranking(hits) -> list[str]:
    seen: list[str] = []
    for hit in hits:
        doc_id = hit.metadata.get("doc_id", hit.id)
        if doc_id not in seen:
            seen.append(doc_id)
    return seen


def _check_docs(case: dict, hits) -> tuple[bool, str]:
    docs = _doc_ranking(hits)
    if case.get("expect_empty"):
        return (not hits), f"devolvió {docs}" if hits else "vacío, como se esperaba"
    problems = []
    if not hits:
        problems.append("no devolvió resultados")
    if "allowed_docs" in case and (extra := [d for d in docs if d not in case["allowed_docs"]]):
        problems.append(f"documentos fuera de lo permitido: {extra}")
    if missing := [d for d in case.get("require_docs", []) if d not in docs]:
        problems.append(f"faltan documentos requeridos: {missing}")
    if bad := [d for d in case.get("forbid_docs", []) if d in docs]:
        problems.append(f"aparecieron documentos prohibidos: {bad}")
    return (not problems), "; ".join(problems) or f"documentos: {docs}"


HASHING_DEFAULT_MIN_SCORE = 0.15


def calibrate(in_scope: dict[str, float], out_scope: dict[str, float]) -> dict:
    """Separabilidad entre el mejor puntaje de consultas CON respuesta y SIN respuesta.

    Si el peor caso con respuesta puntúa más que el mejor caso sin respuesta, cualquier umbral
    intermedio abstiene bien en este set. Se calibra y se evalúa sobre las mismas consultas: mide
    si existe una separación, no cuánto generaliza a documentos o preguntas nuevas.
    """
    cal: dict = {
        "con_respuesta": in_scope,
        "sin_respuesta": out_scope,
        "separable": None,
        "umbral_sugerido": None,
        "margen": None,
    }
    if in_scope:
        scores = list(in_scope.values())
        cal["con_respuesta_min"] = round(min(scores), 3)
        cal["con_respuesta_mediana"] = round(median(scores), 3)
    if out_scope:
        cal["sin_respuesta_max"] = round(max(out_scope.values()), 3)
    if in_scope and out_scope:
        low, high = cal["con_respuesta_min"], cal["sin_respuesta_max"]
        cal["separable"] = high < low
        cal["margen"] = round(low - high, 3)
        if cal["separable"]:
            cal["umbral_sugerido"] = round((low + high) / 2, 3)
    return cal


def _pick_min_score(explicit: float | None, embedder_name: str, cal: dict):
    if explicit is not None:
        return explicit, "--min-score (explícito)"
    if embedder_name.startswith("hashing"):
        return HASHING_DEFAULT_MIN_SCORE, "default del baseline léxico"
    if cal.get("umbral_sugerido") is not None:
        return cal["umbral_sugerido"], "calibrado en este mismo set"
    return None, "sin umbral: no hay separación entre lo relevante y lo irrelevante"


def run(
    corpus: str | Path,
    cases_path: str | Path,
    *,
    embedder_name: str = "hashing",
    k: int = 3,
    thresholds: dict | None = None,
    sensitivity: Sensitivity = "sensitive",
    min_score: float | None = None,
    embedder: Embedder | None = None,
) -> Report:
    docs, _ = ingest_directory(corpus)
    embedder = embedder or build_embedder(embedder_name, sensitivity=sensitivity)
    collection = Collection("evals", embedder, sensitivity=sensitivity)
    collection.sync(docs)

    dense = DenseRetriever(collection)
    retrievers: dict[str, Retriever] = {
        "dense": dense,
        "parent-section": ParentRetriever(dense, collection, "section"),
        "parent-document": ParentRetriever(dense, collection, "document"),
    }
    vocab = vocabulary_from(docs)
    doc_tokens = {d.doc_id: estimate_tokens(d.parents[0].text) for d in docs}
    report = Report(embedder=embedder.name, k=k)
    report.baselines["tokens_corpus_completo"] = float(sum(doc_tokens.values()))

    all_cases = load_cases(cases_path)

    def top_score(query: str) -> float:
        top = dense.retrieve(query, 1)
        return round(top[0].score, 4) if top else 0.0

    in_scope = {c["id"]: top_score(c["query"]) for c in all_cases if c["type"] == "retrieval"}
    out_scope = {c["id"]: top_score(c["query"]) for c in all_cases if c["type"] == "abstain"}
    cal = calibrate(in_scope, out_scope)
    used_min_score, source = _pick_min_score(min_score, embedder.name, cal)
    report.calibration = {**cal, "min_score_usado": used_min_score, "fuente": source}
    if used_min_score is not None:
        # La otra mitad del compromiso: lo que el umbral elegido hace mal en este set.
        report.calibration["falsas_abstenciones"] = sorted(
            i for i, v in in_scope.items() if v < used_min_score
        )
        report.calibration["falsas_respuestas"] = sorted(
            i for i, v in out_scope.items() if v >= used_min_score
        )

    # acumulado[grupo][estrategia] = {"n", "hit", "mrr", "ctx"}
    acc: dict[str, dict[str, dict[str, float]]] = {}
    full_doc_tokens, n_retrieval = 0, 0
    report.misses = {s: [] for s in STRATEGIES}

    for case in all_cases:
        kind, cid = case["type"], case["id"]
        if kind == "retrieval":
            n_retrieval += 1
            group = case.get("group", "léxico")
            expected = set(case["expect_docs"])
            full_doc_tokens += max(doc_tokens[d] for d in expected)
            for name, retriever in retrievers.items():
                hits = retriever.retrieve(case["query"], k)
                ranking = _doc_ranking(hits)
                rank = next((i for i, d in enumerate(ranking, 1) if d in expected), None)
                slot = acc.setdefault(group, {}).setdefault(
                    name, {"n": 0, "hit": 0, "mrr": 0.0, "ctx": 0}
                )
                slot["n"] += 1
                slot["hit"] += rank is not None
                slot["mrr"] += 1.0 / rank if rank else 0.0
                slot["ctx"] += sum(estimate_tokens(h.text) for h in hits)
                if rank is None:
                    report.misses[name].append(cid)
        elif kind == "filter":
            hits = dense.retrieve(case["query"], 10, case.get("where"))
            ok, detail = _check_docs(case, hits)
            report.cases.append(CaseResult(cid, kind, ok, detail))
        elif kind == "selfquery":
            llm = ScriptedLLM(structured=[case["plan"]])
            sq = SelfQueryRetriever(dense, llm, vocabulary=vocab)
            hits = sq.retrieve(case["query"], 10, case.get("scope"))
            ok, detail = _check_docs(case, hits)
            report.cases.append(CaseResult(cid, kind, ok, detail))
        elif kind == "abstain":
            best = out_scope[cid]
            if used_min_score is None:
                detail = f"{source} (mejor puntaje de esta consulta: {best:.3f})"
                report.cases.append(CaseResult(cid, kind, False, detail))
                continue
            llm = ScriptedLLM(texts=["Respuesta inventada [1]"])
            ans = answer(case["query"], dense, llm, k=k, min_score=used_min_score)
            ok = ans.status == "abstained_no_evidence" and not llm.prompts
            detail = (
                f"estado={ans.status}, llamadas al LLM={len(llm.prompts)}, "
                f"mejor puntaje={best:.3f}, umbral={used_min_score:.3f}"
            )
            known = not ok and report.embedder in case.get("known_limit", [])
            report.cases.append(CaseResult(cid, kind, ok, detail, known_limit=known))
        else:
            report.cases.append(CaseResult(cid, kind, False, f"tipo desconocido: {kind}"))

    if n_retrieval:
        def summarize(slot: dict[str, float]) -> dict[str, float]:
            n = slot["n"]
            return {
                "n": n,
                "hit_at_k": round(slot["hit"] / n, 3),
                "mrr": round(slot["mrr"] / n, 3),
                "avg_context_tokens": round(slot["ctx"] / n, 1),
            }

        for group, by_strategy in acc.items():
            report.group_metrics[group] = {s: summarize(v) for s, v in by_strategy.items()}
        for s in STRATEGIES:
            total = {"n": 0, "hit": 0, "mrr": 0.0, "ctx": 0}
            for by_strategy in acc.values():
                for key in total:
                    total[key] += by_strategy[s][key]
            report.metrics[s] = summarize(total)
        report.baselines["avg_tokens_documento_esperado"] = round(full_doc_tokens / n_retrieval, 1)
        report.baselines["n_consultas"] = float(n_retrieval)

    # Umbrales: claves "<grupo>/<estrategia>" -> {métrica: mínimo}, por nombre de embedder.
    gate = (thresholds or {}).get(embedder.name)
    for key, mins in (gate or {}).items():
        group, _, strategy = key.partition("/")
        for metric, floor in mins.items():
            got = report.group_metrics.get(group, {}).get(strategy, {}).get(metric)
            if got is None or got < floor:
                report.threshold_failures.append(f"{key}.{metric}: {got} < mínimo {floor}")
    return report


def to_markdown(r: Report) -> str:
    n = int(r.baselines.get("n_consultas", 0))
    lines = [
        "# Reporte de evals de recuperación",
        "",
        f"Embedder: `{r.embedder}` · k = {r.k} · consultas de recuperación: {n}",
        "",
        "## Recuperación",
        "",
        "`léxico`: la consulta repite palabras del texto. `parafraseo`: usa sinónimos y "
        "otra redacción; separa lo que un baseline léxico resuelve de lo que exige semántica.",
        "",
        "| Grupo | Estrategia | n | hit@k | MRR | Tokens de contexto (prom.) |",
        "|---|---|---|---|---|---|",
    ]
    for group, by_strategy in r.group_metrics.items():
        for s, m in by_strategy.items():
            lines.append(
                f"| {group} | {s} | {int(m['n'])} | {m['hit_at_k']} | {m['mrr']} | "
                f"{m['avg_context_tokens']} |"
            )
    if r.baselines:
        lines += [
            "",
            f"Referencias: el documento esperado completo pesa en promedio "
            f"{r.baselines.get('avg_tokens_documento_esperado')} tokens; todo el corpus, "
            f"{int(r.baselines['tokens_corpus_completo'])} tokens.",
        ]
    for s, missed in r.misses.items():
        if missed:
            lines.append(f"- Fallos de `{s}`: {', '.join(missed)}")
    cal = r.calibration
    if cal:
        lines += ["", "## Calibración de la abstención", ""]
        if cal.get("con_respuesta"):
            lines.append(
                f"- Mejor puntaje de las consultas **con** respuesta: mínimo "
                f"{cal['con_respuesta_min']}, mediana {cal['con_respuesta_mediana']} "
                f"(n = {len(cal['con_respuesta'])})."
            )
        if cal.get("sin_respuesta"):
            detail = ", ".join(f"{i}: {v}" for i, v in cal["sin_respuesta"].items())
            lines.append(
                f"- Mejor puntaje de las consultas **sin** respuesta: máximo "
                f"{cal['sin_respuesta_max']} ({detail})."
            )
        if cal.get("separable") is not None:
            verdict = f"sí, con margen {cal['margen']}" if cal["separable"] else (
                f"NO (superposición de {-cal['margen']}): ningún umbral abstiene bien en este set"
            )
            lines.append(f"- ¿Se separan?: {verdict}. Umbral sugerido: {cal['umbral_sugerido']}.")
        lines.append(f"- Umbral usado: {cal['min_score_usado']} ({cal['fuente']}).")
        if cal.get("min_score_usado") is not None:
            fa, fr = cal["falsas_abstenciones"], cal["falsas_respuestas"]
            wrong_abstain = f" ({', '.join(fa)})" if fa else ""
            wrong_answer = f" ({', '.join(fr)})" if fr else ""
            lines.append(
                f"- Con ese umbral: se abstendría indebidamente en {len(fa)} de "
                f"{len(cal['con_respuesta'])} consultas con respuesta{wrong_abstain} y "
                f"respondería indebidamente en {len(fr)} de {len(cal['sin_respuesta'])} sin "
                f"respuesta{wrong_answer}."
            )
        lines.append(
            "- Está calibrado y evaluado sobre las mismas consultas: confirma que existe una "
            "separación, no que generalice. Para tu corpus real armá tu propio set de preguntas "
            "sin respuesta y calibrá con él."
        )
    lines += [
        "",
        "## Casos deterministas",
        "",
        "| Caso | Tipo | Resultado | Detalle |",
        "|---|---|---|---|",
    ]
    for c in r.cases:
        label = "OK" if c.passed else ("LÍMITE CONOCIDO" if c.known_limit else "FALLA")
        lines.append(f"| {c.id} | {c.type} | {label} | {c.detail} |")
    if r.threshold_failures:
        lines += ["", "## Umbrales incumplidos", ""] + [f"- {t}" for t in r.threshold_failures]
    lines += [
        "",
        "Notas: tokens estimados (~4 caracteres/token). Corpus sintético y ficticio. Con "
        "`hashing` el modelo es un baseline léxico: no mide comprensión semántica. Los casos "
        "`selfquery` usan un LLM de guion y verifican validación y alcance, no la calidad de un "
        "modelo real.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evals de recuperación del pipeline RAG.")
    parser.add_argument("--corpus", default="data/synthetic_corpus")
    parser.add_argument("--cases", default="evals/cases.jsonl")
    parser.add_argument("--embedder", choices=["hashing", "local", "watsonx"], default="hashing")
    parser.add_argument(
        "--sensitivity", choices=["sensitive", "public"], default="sensitive",
        help="watsonx exige 'public' (el texto sale a la nube); válido con el corpus sintético",
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--min-score", type=float, default=None,
        help="umbral de abstención; por defecto se calibra con las consultas del set",
    )
    parser.add_argument("--thresholds", default="evals/thresholds.json")
    parser.add_argument("--out-md")
    parser.add_argument("--out-json")
    args = parser.parse_args(argv)

    thresholds = None
    if args.thresholds and Path(args.thresholds).exists():
        thresholds = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    report = run(
        args.corpus, args.cases, embedder_name=args.embedder, k=args.k,
        thresholds=thresholds, sensitivity=args.sensitivity, min_score=args.min_score,
    )
    markdown = to_markdown(report)
    print(markdown)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(markdown, encoding="utf-8")
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
