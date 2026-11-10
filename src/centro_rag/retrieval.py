"""Estrategias de recuperación componibles.

Todas implementan `retrieve(query, k, where) -> list[Hit]` y se pueden apilar:

    SelfQueryRetriever(MultiQueryRetriever(DenseRetriever(col), llm), llm)

* `DenseRetriever`      búsqueda por similitud sobre chunks.
* `ParentRetriever`     busca en chunks chicos pero devuelve su sección o su documento completo.
* `MultiQueryRetriever` reformula la consulta con un LLM y fusiona rankings con RRF.
* `SelfQueryRetriever`  el LLM propone filtros por metadata; se validan contra una lista blanca.

El `where` que pasa quien llama es un ALCANCE OBLIGATORIO (por ejemplo, permisos): el LLM solo
puede agregar restricciones, nunca quitar las que ya vienen.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, ValidationError

from centro_rag.llm import LLM, StructuredOutputError
from centro_rag.store import Clause, Collection, Hit, normalize_where


class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 4, where: dict | list | None = None) -> list[Hit]: ...


class DenseRetriever:
    def __init__(self, collection: Collection) -> None:
        self.collection = collection

    def retrieve(self, query: str, k: int = 4, where: dict | list | None = None) -> list[Hit]:
        return self.collection.search(query, k, where)


# --------------------------------------------------------------------------- padre
class ParentRetriever:
    """Devuelve la sección (o el documento completo) a la que pertenece cada chunk encontrado.

    `level="document"` devuelve el documento entero: puede ser grande. Si preferís acotar el
    contexto, usá `level="section"` (por defecto).
    """

    def __init__(
        self,
        base: Retriever,
        collection: Collection,
        level: Literal["section", "document"] = "section",
        overfetch: int = 4,
    ) -> None:
        if overfetch < 1:
            raise ValueError("overfetch debe ser >= 1")
        self.base, self.collection, self.level, self.overfetch = base, collection, level, overfetch

    def retrieve(self, query: str, k: int = 4, where: dict | list | None = None) -> list[Hit]:
        field = "section_id" if self.level == "section" else "doc_id"
        children = self.base.retrieve(query, k * self.overfetch, where)
        best: dict[str, Hit] = {}
        matched: dict[str, int] = defaultdict(int)
        for hit in children:  # ya vienen ordenados por relevancia
            key = hit.metadata.get(field)
            if key is None:
                continue
            best.setdefault(key, hit)
            matched[key] += 1
        keys = list(best)[:k]
        texts = self.collection.parents(keys)
        out = []
        for key in keys:
            child = best[key]
            meta = {**child.metadata, "matched_chunks": matched[key], "parent_level": self.level}
            if key in texts:
                out.append(Hit(key, texts[key], meta, child.score))
            else:  # padre ausente (índice inconsistente): se degrada al chunk y se avisa
                out.append(Hit(child.id, child.text, {**meta, "parent_missing": True}, child.score))
        return out


# --------------------------------------------------------------------------- multi-query
def _norm(text: str) -> str:
    """Clave de comparación: sin mayúsculas, signos de puntuación ni espacios repetidos."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def parse_variants(text: str, limit: int) -> list[str]:
    """Una reformulación por línea; quita viñetas/numeración y duplicados (ignora puntuación)."""
    seen: set[str] = set()
    out: list[str] = []
    for line in text.splitlines():
        clean = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"')
        key = _norm(clean)
        if clean and key not in seen:
            seen.add(key)
            out.append(clean)
    return out[:limit]


class MultiQueryRetriever:
    """Reformula la consulta con un LLM y fusiona los rankings con Reciprocal Rank Fusion."""

    def __init__(
        self, base: Retriever, llm: LLM, n_variants: int = 3, rrf_k: int = 60
    ) -> None:
        if n_variants < 1 or rrf_k < 1:
            raise ValueError("n_variants y rrf_k deben ser >= 1")
        self.base, self.llm, self.n_variants, self.rrf_k = base, llm, n_variants, rrf_k
        self.last_queries: list[str] = []

    def variants(self, query: str) -> list[str]:
        prompt = (
            f"Escribí {self.n_variants} reformulaciones distintas de la pregunta para buscar "
            "en una base documental. Una por línea, sin numeración ni comentarios.\n"
            f"Pregunta: {query}"
        )
        extra = [
            v for v in parse_variants(self.llm.generate(prompt), self.n_variants)
            if _norm(v) != _norm(query)
        ]
        return [query, *extra]

    def retrieve(self, query: str, k: int = 4, where: dict | list | None = None) -> list[Hit]:
        self.last_queries = self.variants(query)
        fused: dict[str, float] = defaultdict(float)
        first: dict[str, Hit] = {}
        best_cos: dict[str, float] = {}
        for q in self.last_queries:
            for rank, hit in enumerate(self.base.retrieve(q, k * 2, where), start=1):
                fused[hit.id] += 1.0 / (self.rrf_k + rank)
                first.setdefault(hit.id, hit)
                best_cos[hit.id] = max(best_cos.get(hit.id, hit.score), hit.score)
        ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [
            Hit(i, first[i].text, {**first[i].metadata, "rrf": round(s, 6)}, best_cos[i])
            for i, s in ranked
        ]


# --------------------------------------------------------------------------- self-query
@dataclass(frozen=True)
class FieldSpec:
    type: Literal["int", "str"]
    description: str


DEFAULT_FIELDS: dict[str, FieldSpec] = {
    "year": FieldSpec("int", "Año de publicación del documento"),
    "institution": FieldSpec("str", "Institución emisora"),
    "document_type": FieldSpec("str", "Tipo de documento (protocolo, normativa, informe, manual)"),
}


class FilterClause(BaseModel):
    field: str
    op: Literal["eq", "gt", "gte", "lt", "lte", "in"] = "eq"
    value: Any


class QueryPlan(BaseModel):
    query: str = Field(min_length=1, description="Consulta sin las condiciones de metadata")
    filters: list[FilterClause] = Field(default_factory=list)


class PlanError(ValueError):
    """El plan propuesto por el LLM no cumple la lista blanca."""


def validate_plan(
    plan: QueryPlan,
    fields: dict[str, FieldSpec],
    vocabulary: dict[str, list[str]] | None = None,
) -> list[Clause]:
    """Convierte el plan en cláusulas seguras o lanza `PlanError` con un mensaje accionable."""
    clauses: list[Clause] = []
    vocab = {f: {v.lower(): v for v in vals} for f, vals in (vocabulary or {}).items()}

    def check(spec: FieldSpec, name: str, value: Any) -> Any:
        if spec.type == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                raise PlanError(f"'{name}' requiere un entero, vino {value!r}")
            return value
        if not isinstance(value, str):
            raise PlanError(f"'{name}' requiere texto, vino {value!r}")
        if name in vocab:
            canon = vocab[name].get(value.strip().lower())
            if canon is None:
                raise PlanError(
                    f"'{value}' no es un valor válido de '{name}'. Válidos: "
                    f"{', '.join(sorted(vocab[name].values()))}"
                )
            return canon
        return value.strip()

    for fc in plan.filters:
        spec = fields.get(fc.field)
        if spec is None:
            raise PlanError(f"Campo no permitido: {fc.field!r}. Permitidos: {sorted(fields)}")
        if fc.op in ("gt", "gte", "lt", "lte") and spec.type != "int":
            raise PlanError(f"El operador {fc.op!r} solo aplica a campos enteros")
        if fc.op == "in":
            if not isinstance(fc.value, list) or not fc.value:
                raise PlanError("El operador 'in' requiere una lista no vacía")
            value: Any = [check(spec, fc.field, v) for v in fc.value]
        else:
            value = check(spec, fc.field, fc.value)
        clauses.append((fc.field, fc.op, value))
    return clauses


class SelfQueryRetriever:
    """El LLM separa la consulta de las condiciones de metadata; solo pasan filtros válidos."""

    def __init__(
        self,
        base: Retriever,
        llm: LLM,
        fields: dict[str, FieldSpec] | None = None,
        vocabulary: dict[str, list[str]] | None = None,
        max_attempts: int = 3,
    ) -> None:
        self.base, self.llm = base, llm
        self.fields = fields or DEFAULT_FIELDS
        self.vocabulary = vocabulary
        self.max_attempts = max_attempts
        self.last_plan: QueryPlan | None = None

    def _prompt(self, question: str) -> str:
        lines = [f"- {n} ({s.type}): {s.description}" for n, s in self.fields.items()]
        if self.vocabulary:
            lines += [f"  valores de {f}: {', '.join(v)}" for f, v in self.vocabulary.items()]
        return (
            "Convertí la pregunta en un plan de búsqueda: `query` (sin las condiciones) y "
            "`filters` solo con los campos listados. Operadores: eq, gt, gte, lt, lte, in.\n"
            "Campos:\n" + "\n".join(lines) + f"\nPregunta: {question}"
        )

    def plan(self, question: str) -> tuple[QueryPlan, list[Clause]]:
        prompt, last = self._prompt(question), ""
        for _ in range(self.max_attempts):
            try:
                plan = self.llm.structured(prompt, QueryPlan)
                return plan, validate_plan(plan, self.fields, self.vocabulary)
            except (ValidationError, PlanError) as exc:
                last = str(exc).replace("\n", " ")[:300]
                prompt = f"{self._prompt(question)}\n\nTu plan anterior falló: {last}\nCorregilo."
        raise StructuredOutputError(f"Plan inválido tras {self.max_attempts} intentos: {last}")

    def retrieve(self, query: str, k: int = 4, where: dict | list | None = None) -> list[Hit]:
        plan, clauses = self.plan(query)
        self.last_plan = plan
        scope = normalize_where(where)  # el alcance del llamador siempre se conserva
        return self.base.retrieve(plan.query, k, scope + clauses)
