"""CLI mínima: `rag ingest` indexa un directorio y `rag search` consulta (sin generación)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from centro_rag.embeddings import build_embedder
from centro_rag.ingest import IngestError, ingest_directory
from centro_rag.retrieval import DenseRetriever, ParentRetriever
from centro_rag.store import Collection, MemoryBackend

_COND = re.compile(r"^(\w+)(>=|<=|>|<|=)(.+)$")
_OPS = {"=": "eq", ">": "gt", ">=": "gte", "<": "lt", "<=": "lte"}


def parse_where(items: list[str] | None) -> dict | None:
    """`year>=2022` `institution=Hospital Modelo` -> filtros normalizados."""
    if not items:
        return None
    clauses = []
    for item in items:
        m = _COND.match(item.strip())
        if not m:
            raise ValueError(f"Filtro inválido: {item!r} (formato campo=valor, campo>=valor, ...)")
        fld, op, raw = m.groups()
        value: int | str = int(raw) if fld == "year" else raw.strip()
        clauses.append((fld, _OPS[op], value))
    return clauses  # type: ignore[return-value]


def _collection(args: argparse.Namespace) -> Collection:
    embedder = build_embedder(args.embedder, sensitivity=args.sensitivity)
    store = Path(args.store)
    if args.backend == "chroma":
        from centro_rag.store import ChromaBackend

        backend = ChromaBackend(store, args.name)
    else:
        backend = MemoryBackend(store / f"{args.name}.json")
    return Collection(args.name, embedder, backend, sensitivity=args.sensitivity)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("ingest", "search"):
        p = sub.add_parser(name)
        p.add_argument("--store", default=".rag_store", help="directorio de la colección")
        p.add_argument("--name", default="default")
        p.add_argument("--backend", choices=["json", "chroma"], default="json")
        p.add_argument("--embedder", choices=["hashing", "local", "watsonx"], default="local")
        p.add_argument("--sensitivity", choices=["sensitive", "public"], default="sensitive")
    ing = sub.choices["ingest"]
    ing.add_argument("--dir", required=True)
    ing.add_argument("--prune", action="store_true", help="borra lo que ya no está en --dir")
    sea = sub.choices["search"]
    sea.add_argument("query")
    sea.add_argument("-k", type=int, default=4)
    sea.add_argument("--where", action="append", help="p. ej. year>=2022 (repetible)")
    sea.add_argument("--level", choices=["chunk", "section", "document"], default="chunk")

    args = parser.parse_args(argv)
    try:
        col = _collection(args)
        if args.cmd == "ingest":
            docs, skipped = ingest_directory(args.dir)
            rep = col.sync(docs, prune=args.prune)
            print(f"agregados={len(rep.added)} actualizados={len(rep.updated)} "
                  f"sin_cambios={len(rep.unchanged)} borrados={len(rep.deleted)} "
                  f"omitidos={len(skipped)} chunks={col.count()}")
            return 0
        retriever = DenseRetriever(col)
        if args.level != "chunk":
            retriever = ParentRetriever(retriever, col, args.level)
        for i, hit in enumerate(retriever.retrieve(args.query, args.k, parse_where(args.where)), 1):
            snippet = " ".join(hit.text.split())[:140]
            print(f"[{i}] {hit.score:.3f} {hit.metadata.get('source', hit.id)} :: {snippet}")
        return 0
    except (IngestError, ValueError, RuntimeError, PermissionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
