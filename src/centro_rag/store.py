"""Almacenamiento vectorial con aislamiento, upsert/borrado reales y filtros por metadata.

Dos backends con la MISMA interfaz y las mismas pruebas:
* `MemoryBackend`: numpy, con persistencia JSON opcional. Sirve para tests y corpus chicos.
* `ChromaBackend`: ChromaDB persistente (extra `chroma`).

Empates: los resultados con el mismo puntaje (típicamente 0, sin ningún término en común con la
consulta) NO tienen orden garantizado entre backends. Usá un puntaje mínimo (`qa.answer` lo hace)
en lugar de confiar en el orden de la cola del ranking.

Aislamiento: cada `Collection` tiene su propio backend. Dos colecciones NUNCA comparten datos
(en Chroma, nombres de colección distintos dentro del mismo directorio). La versión anterior
usaba el cliente efímero por defecto, que comparte estado entre instancias del mismo proceso.
"""

from __future__ import annotations

import json
import math
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from centro_rag.embeddings import CloudNotAllowedError, Embedder, Sensitivity
from centro_rag.ingest import IngestedDoc

Clause = tuple[str, str, Any]  # (campo, operador, valor)
OPS = ("eq", "gt", "gte", "lt", "lte", "in")
_META_KEY = "__meta__"


class EmbedderMismatchError(RuntimeError):
    """La colección se creó con otro embedder o con otra política de sensibilidad."""


@dataclass(frozen=True)
class Hit:
    id: str
    text: str
    metadata: dict[str, Any]
    score: float  # similitud coseno (1 = idéntico), también tras fusionar consultas


# --------------------------------------------------------------------------- filtros
def normalize_where(where: dict | list | None) -> list[Clause]:
    """`{"year": {"$gte": 2022}, "institution": "X"}` -> lista de cláusulas validadas."""
    if not where:
        return []
    if isinstance(where, list):
        clauses = [(f, o.lstrip("$"), v) for f, o, v in where]
    else:
        clauses = []
        for fld, cond in where.items():
            if isinstance(cond, dict):
                clauses += [(fld, op.lstrip("$"), val) for op, val in cond.items()]
            else:
                clauses.append((fld, "eq", cond))
    for fld, op, val in clauses:
        if op not in OPS:
            raise ValueError(f"Operador no soportado: {op!r} (válidos: {', '.join(OPS)})")
        if op == "in" and not isinstance(val, list | tuple | set):
            raise ValueError("El operador 'in' requiere una lista")
        if not isinstance(fld, str) or not fld:
            raise ValueError("Nombre de campo inválido")
    return clauses


_COMPARE = {
    "eq": operator.eq,
    "gt": operator.gt,
    "gte": operator.ge,
    "lt": operator.lt,
    "lte": operator.le,
    "in": lambda have, options: have in options,
}


def matches(meta: dict[str, Any], clauses: list[Clause]) -> bool:
    for fld, op, val in clauses:
        if fld not in meta:
            return False
        try:
            ok = _COMPARE[op](meta[fld], val)
        except TypeError:  # tipos incomparables (p. ej. str vs int): no coincide
            ok = False
        if not ok:
            return False
    return True


def _chroma_where(clauses: list[Clause]) -> dict | None:
    if not clauses:
        return None
    parts = [{f: {f"${op}": (list(v) if op == "in" else v)}} for f, op, v in clauses]
    return parts[0] if len(parts) == 1 else {"$and": parts}


def _ids_for_doc(key: str, doc_id: str) -> bool:
    return key == doc_id or key.startswith(doc_id + "#")


# --------------------------------------------------------------------------- backends
class Backend(Protocol):
    def get_meta(self) -> dict: ...
    def set_meta(self, meta: dict) -> None: ...
    def upsert_chunks(
        self, ids: list[str], embeddings: list[list[float]], texts: list[str],
        metadatas: list[dict],
    ) -> None: ...
    def put_parents(self, parents: dict[str, str], doc_id: str) -> None: ...
    def get_parents(self, keys: list[str]) -> dict[str, str]: ...
    def delete_doc(self, doc_id: str) -> int: ...
    def query(self, embedding: list[float], k: int, clauses: list[Clause]) -> list[Hit]: ...
    def doc_hashes(self) -> dict[str, str]: ...
    def count(self) -> int: ...
    def flush(self) -> None: ...


class MemoryBackend:
    """Búsqueda exacta por fuerza bruta con numpy: O(n) por consulta (ok para miles de chunks)."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else None
        self._chunks: dict[str, dict] = {}
        self._parents: dict[str, tuple[str, str]] = {}  # key -> (doc_id, texto)
        self._meta: dict = {}
        self._cache: tuple[list[str], np.ndarray] | None = None
        if self._path and self._path.exists():
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._chunks, self._meta = data["chunks"], data["meta"]
            self._parents = {k: tuple(v) for k, v in data["parents"].items()}

    def get_meta(self) -> dict:
        return dict(self._meta)

    def set_meta(self, meta: dict) -> None:
        self._meta = dict(meta)

    def upsert_chunks(self, ids, embeddings, texts, metadatas) -> None:
        for i, e, t, m in zip(ids, embeddings, texts, metadatas, strict=True):
            self._chunks[i] = {"emb": list(e), "text": t, "meta": dict(m)}
        self._cache = None

    def put_parents(self, parents: dict[str, str], doc_id: str) -> None:
        for key, text in parents.items():
            self._parents[key] = (doc_id, text)

    def get_parents(self, keys: list[str]) -> dict[str, str]:
        return {k: self._parents[k][1] for k in keys if k in self._parents}

    def delete_doc(self, doc_id: str) -> int:
        gone = [i for i, c in self._chunks.items() if c["meta"].get("doc_id") == doc_id]
        for i in gone:
            del self._chunks[i]
        for k in [k for k, (d, _) in self._parents.items() if d == doc_id]:
            del self._parents[k]
        self._cache = None
        return len(gone)

    def _matrix(self) -> tuple[list[str], np.ndarray]:
        if self._cache is None:
            ids = sorted(self._chunks)
            mat = np.array([self._chunks[i]["emb"] for i in ids], dtype=np.float64)
            self._cache = (ids, mat.reshape(len(ids), -1))
        return self._cache

    def query(self, embedding, k, clauses) -> list[Hit]:
        if k < 1 or not self._chunks:
            return []
        ids, mat = self._matrix()
        q = np.asarray(embedding, dtype=np.float64)
        norms = np.linalg.norm(mat, axis=1) * (np.linalg.norm(q) or 1.0)
        norms[norms == 0] = 1.0
        scores = (mat @ q) / norms
        ranked = sorted(
            (
                (-round(float(s), 9), i)
                for i, s in zip(ids, scores, strict=True)
                if matches(self._chunks[i]["meta"], clauses)
            )
        )[:k]
        return [
            Hit(i, self._chunks[i]["text"], dict(self._chunks[i]["meta"]), -s) for s, i in ranked
        ]

    def doc_hashes(self) -> dict[str, str]:
        return {
            c["meta"]["doc_id"]: c["meta"]["content_hash"]
            for c in self._chunks.values()
            if "doc_id" in c["meta"] and "content_hash" in c["meta"]
        }

    def count(self) -> int:
        return len(self._chunks)

    def flush(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"chunks": self._chunks, "parents": self._parents, "meta": self._meta}
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._path)  # escritura atómica


class ChromaBackend:
    """ChromaDB persistente. Embeddings siempre explícitos: nunca se descarga un modelo oculto."""

    def __init__(self, path: str | Path, name: str = "default") -> None:
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:  # pragma: no cover - depende del entorno
            raise RuntimeError("Falta el extra: pip install '.[chroma]'") from exc
        self._client = chromadb.PersistentClient(
            path=str(path), settings=Settings(anonymized_telemetry=False)
        )
        self._chunks = self._client.get_or_create_collection(
            f"{name}__chunks", embedding_function=None, metadata={"hnsw:space": "cosine"}
        )
        self._parents = self._client.get_or_create_collection(
            f"{name}__parents", embedding_function=None
        )

    def get_meta(self) -> dict:
        got = self._parents.get(ids=[_META_KEY])
        return json.loads(got["documents"][0]) if got["ids"] else {}

    def set_meta(self, meta: dict) -> None:
        self._parents.upsert(
            ids=[_META_KEY], documents=[json.dumps(meta)], embeddings=[[0.0]],
            metadatas=[{"doc_id": _META_KEY}],
        )

    def upsert_chunks(self, ids, embeddings, texts, metadatas) -> None:
        if ids:
            self._chunks.upsert(
                ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas
            )

    def put_parents(self, parents: dict[str, str], doc_id: str) -> None:
        if parents:
            keys = list(parents)
            self._parents.upsert(
                ids=keys, documents=[parents[k] for k in keys],
                embeddings=[[0.0]] * len(keys), metadatas=[{"doc_id": doc_id}] * len(keys),
            )

    def get_parents(self, keys: list[str]) -> dict[str, str]:
        if not keys:
            return {}
        got = self._parents.get(ids=keys)
        return dict(zip(got["ids"], got["documents"], strict=True))

    def delete_doc(self, doc_id: str) -> int:
        before = len(self._chunks.get(where={"doc_id": doc_id})["ids"])
        self._chunks.delete(where={"doc_id": doc_id})
        self._parents.delete(where={"doc_id": doc_id})
        return before

    def query(self, embedding, k, clauses) -> list[Hit]:
        total = self._chunks.count()
        if k < 1 or total == 0:
            return []
        res = self._chunks.query(
            query_embeddings=[list(embedding)], n_results=min(k, total),
            where=_chroma_where(clauses),
        )
        return [
            Hit(i, doc, dict(meta), 1.0 - float(dist))
            for i, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0],
                strict=True,
            )
        ]

    def doc_hashes(self) -> dict[str, str]:
        got = self._chunks.get(include=["metadatas"])
        return {
            m["doc_id"]: m["content_hash"]
            for m in got["metadatas"]
            if m and "doc_id" in m and "content_hash" in m
        }

    def count(self) -> int:
        return self._chunks.count()

    def flush(self) -> None:  # Chroma persiste por su cuenta
        return None


# --------------------------------------------------------------------------- colección
@dataclass
class SyncReport:
    added: list[str]
    updated: list[str]
    unchanged: list[str]
    deleted: list[str]


class Collection:
    """Colección lógica: un embedder, un backend y una política de sensibilidad."""

    def __init__(
        self,
        name: str,
        embedder: Embedder,
        backend: Backend | None = None,
        *,
        sensitivity: Sensitivity = "sensitive",
    ) -> None:
        if embedder.is_cloud and sensitivity != "public":
            raise CloudNotAllowedError(
                "Embedder en la nube con datos 'sensitive': declará sensitivity='public' "
                "solo si el contenido puede salir del servidor."
            )
        self.name = name
        self.embedder = embedder
        self.backend: Backend = backend or MemoryBackend()
        stored = self.backend.get_meta()
        if stored:
            if stored.get("embedder") != embedder.name:
                raise EmbedderMismatchError(
                    f"La colección usa {stored.get('embedder')!r}; vino {embedder.name!r}. "
                    "Mezclar espacios vectoriales rompe la búsqueda: reindexá en otra colección."
                )
            if stored.get("sensitivity") != sensitivity:
                raise EmbedderMismatchError(
                    f"La colección fue creada como {stored.get('sensitivity')!r}, "
                    f"no como {sensitivity!r}."
                )
        else:
            self.backend.set_meta(
                {"name": name, "embedder": embedder.name, "sensitivity": sensitivity}
            )
            self.backend.flush()

    # ---- escritura
    def upsert_document(self, doc: IngestedDoc) -> Literal["added", "updated", "unchanged"]:
        old = self.backend.doc_hashes().get(doc.doc_id)
        if old == doc.content_hash:
            return "unchanged"
        if old is not None:
            self.backend.delete_doc(doc.doc_id)  # sin chunks huérfanos si el doc encogió
        if doc.chunks:
            embeddings = self.embedder.embed_documents([c.embed_text for c in doc.chunks])
            if len(embeddings) != len(doc.chunks) or any(
                not all(math.isfinite(x) for x in e) for e in embeddings
            ):
                raise RuntimeError("El embedder devolvió vectores inválidos")
            self.backend.upsert_chunks(
                [c.id for c in doc.chunks], embeddings, [c.text for c in doc.chunks],
                [c.metadata for c in doc.chunks],
            )
        self.backend.put_parents({p.key: p.text for p in doc.parents}, doc.doc_id)
        self.backend.flush()
        return "updated" if old is not None else "added"

    def delete_document(self, doc_id: str) -> int:
        """Borra un documento y sus padres. Devuelve cuántos chunks se eliminaron."""
        removed = self.backend.delete_doc(doc_id)
        self.backend.flush()
        return removed

    def sync(self, docs: list[IngestedDoc], *, prune: bool = False) -> SyncReport:
        """Alinea la colección con `docs`. Con `prune=True` borra lo que ya no está en la fuente."""
        report = SyncReport([], [], [], [])
        for doc in docs:
            getattr(report, {"added": "added", "updated": "updated", "unchanged": "unchanged"}[
                self.upsert_document(doc)
            ]).append(doc.doc_id)
        if prune:
            keep = {d.doc_id for d in docs}
            for doc_id in sorted(set(self.backend.doc_hashes()) - keep):
                self.delete_document(doc_id)
                report.deleted.append(doc_id)
        return report

    # ---- lectura
    def search(self, query: str, k: int = 4, where: dict | list | None = None) -> list[Hit]:
        return self.backend.query(self.embedder.embed_query(query), k, normalize_where(where))

    def parents(self, keys: list[str]) -> dict[str, str]:
        return self.backend.get_parents(keys)

    def doc_ids(self) -> list[str]:
        return sorted(self.backend.doc_hashes())

    def count(self) -> int:
        return self.backend.count()
