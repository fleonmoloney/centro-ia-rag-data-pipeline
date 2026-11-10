"""Ingesta: lectura de archivos, metadata, chunking jerárquico e identificadores estables.

Decisiones que importan:
* `doc_id` estable derivado de la ruta relativa -> re-ingestar el mismo archivo hace upsert,
  no duplica.
* `content_hash` -> permite saltar documentos sin cambios.
* Metadata filtrable (`year`, `institution`, `document_type`) desde front matter (`---`) en
  .md/.txt o desde un sidecar `<archivo>.meta.json`. Sin metadata no hay self-query posible.
* Un PDF sin texto extraíble (escaneado) falla con un error claro: no hay OCR en este repo.
* Cada chunk lleva su `section_id` y su `doc_id`: son las claves para recuperar el contexto
  "padre" (sección o documento completo).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

FILTERABLE_FIELDS = ("year", "institution", "document_type")
SUPPORTED_SUFFIXES = (".md", ".txt", ".pdf", ".csv", ".json")
_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
_HASH_VERSION = "ingest-v1"


class IngestError(RuntimeError):
    """No se pudo ingerir un archivo."""


class NoTextError(IngestError):
    """El archivo no tiene texto extraíble (típico de un PDF escaneado)."""


@dataclass(frozen=True)
class ChunkConfig:
    chunk_size: int = 600
    chunk_overlap: int = 80

    def __post_init__(self) -> None:
        if self.chunk_size < 50 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_size >= 50 y 0 <= chunk_overlap < chunk_size")


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    context: str  # ruta de headers; se antepone al texto SOLO para calcular el embedding
    metadata: dict[str, Any]

    @property
    def embed_text(self) -> str:
        return f"{self.context}\n{self.text}" if self.context else self.text


@dataclass
class Parent:
    key: str  # `doc_id` (documento completo) o `doc_id#sN` (sección)
    doc_id: str
    level: str  # "document" | "section"
    text: str


@dataclass
class IngestedDoc:
    doc_id: str
    source: str
    content_hash: str
    metadata: dict[str, Any]
    chunks: list[Chunk] = field(default_factory=list)
    parents: list[Parent] = field(default_factory=list)


@dataclass
class _Section:
    text: str
    headers: dict[str, str] = field(default_factory=dict)
    page: int | None = None


# --------------------------------------------------------------------------- utilidades
def slugify(value: str) -> str:
    """Minúsculas, solo [a-z0-9] separados por '-'. Seguro como parte de un identificador."""
    import unicodedata

    folded = unicodedata.normalize("NFKD", value.lower())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return "-".join(re.findall(r"[a-z0-9]+", folded)) or "doc"


def _coerce(key: str, value: Any) -> Any:
    if key == "year":
        try:
            return int(str(value).strip())
        except ValueError as exc:
            raise IngestError(f"'year' debe ser entero, vino {value!r}") from exc
    return str(value).strip() if isinstance(value, str) else value


def parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Front matter mínimo `---\\nclave: valor\\n---`. Devuelve (metadata, cuerpo)."""
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n?", text, flags=re.DOTALL)
    if not match:
        return {}, text
    meta: dict[str, Any] = {}
    for line in match.group(1).splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        if ":" not in line:
            raise IngestError(f"Línea inválida en front matter: {line!r}")
        key, _, value = line.partition(":")
        meta[key.strip()] = _coerce(key.strip(), value)
    return meta, text[match.end():]


def _sidecar(path: Path) -> dict[str, Any]:
    side = path.with_name(path.name + ".meta.json")
    if not side.exists():
        return {}
    try:
        data = json.loads(side.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IngestError(f"{side.name}: JSON inválido ({exc})") from exc
    return {k: _coerce(k, v) for k, v in data.items()}


def _clean_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Chroma solo admite escalares; descartamos None y estructuras."""
    return {
        k: v
        for k, v in meta.items()
        if v is not None and isinstance(v, str | int | float | bool)
    }


# --------------------------------------------------------------------------- lectura
def _read_pdf(path: Path) -> list[_Section]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
        pages = [(i + 1, (p.extract_text() or "").strip()) for i, p in enumerate(reader.pages)]
    except Exception as exc:  # pypdf lanza varios tipos según el daño del archivo
        raise IngestError(f"{path.name}: no se pudo leer el PDF ({type(exc).__name__})") from exc
    sections = [_Section(text=t, page=n) for n, t in pages if t]
    if not sections:
        raise NoTextError(
            f"{path.name}: el PDF no tiene texto extraíble (¿escaneado?). "
            "Este repo no hace OCR; procesalo antes con una herramienta de OCR."
        )
    return sections


def _read_sections(path: Path) -> tuple[list[_Section], dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path), {}
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise IngestError(f"{path.name}: no es UTF-8 válido") from exc
    if suffix in (".md", ".txt"):
        meta, body = parse_front_matter(raw)
        if suffix == ".md":
            docs = MarkdownHeaderTextSplitter(_HEADERS, strip_headers=False).split_text(body)
            sections = [_Section(d.page_content, dict(d.metadata)) for d in docs]
        else:
            sections = [_Section(body.strip())]
        return [s for s in sections if s.text.strip()], meta
    if suffix == ".csv":
        rows = csv.DictReader(io.StringIO(raw))
        lines = [" | ".join(f"{k}: {v}" for k, v in row.items()) for row in rows]
        return [_Section("\n".join(lines))], {}
    if suffix == ".json":
        try:
            pretty = json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
        except json.JSONDecodeError as exc:
            raise IngestError(f"{path.name}: JSON inválido ({exc})") from exc
        return [_Section(pretty)], {}
    raise IngestError(f"{path.name}: formato no soportado ({suffix})")


# --------------------------------------------------------------------------- ingesta
def ingest_file(
    path: str | Path,
    *,
    root: str | Path | None = None,
    config: ChunkConfig | None = None,
    metadata: dict[str, Any] | None = None,
) -> IngestedDoc:
    """Lee un archivo y devuelve el documento listo para indexar (sin tocar ningún store)."""
    path = Path(path)
    config = config or ChunkConfig()
    if not path.is_file():
        raise IngestError(f"No existe el archivo: {path}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise IngestError(f"{path.name}: formato no soportado ({path.suffix or 'sin extensión'})")

    rel = path.relative_to(root) if root else Path(path.name)
    doc_id = slugify(str(rel.with_suffix("")))

    sections, front = _read_sections(path)
    if not sections:
        raise NoTextError(f"{path.name}: el archivo no tiene contenido")
    doc_meta = _clean_meta({**front, **_sidecar(path), **(metadata or {})})

    digest = hashlib.sha256(
        json.dumps(
            [_HASH_VERSION, config.chunk_size, config.chunk_overlap, doc_meta,
             [(s.text, s.headers, s.page) for s in sections]],
            sort_keys=True, ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
    )
    doc = IngestedDoc(doc_id, str(rel), digest, doc_meta)
    doc.parents.append(
        Parent(doc_id, doc_id, "document", "\n\n".join(s.text for s in sections))
    )

    n = 0
    for i, section in enumerate(sections):
        section_id = f"{doc_id}#s{i}"
        doc.parents.append(Parent(section_id, doc_id, "section", section.text))
        context = " > ".join(v for _, v in sorted(section.headers.items()))
        for piece in splitter.split_text(section.text):
            if not piece.strip():
                continue
            meta = {
                **doc_meta,
                **{k: v for k, v in section.headers.items()},
                "doc_id": doc_id,
                "section_id": section_id,
                "chunk_index": n,
                "source": str(rel),
                "content_hash": digest,
            }
            if section.page is not None:
                meta["page"] = section.page
            doc.chunks.append(Chunk(f"{doc_id}:{n}", doc_id, piece, context, _clean_meta(meta)))
            n += 1
    return doc


def ingest_directory(
    root: str | Path, *, config: ChunkConfig | None = None
) -> tuple[list[IngestedDoc], list[str]]:
    """Ingiere recursivamente. Devuelve (documentos, archivos_omitidos).

    Falla si dos archivos producen el mismo `doc_id` (evita sobrescritura silenciosa).
    """
    root = Path(root)
    if not root.is_dir():
        raise IngestError(f"No existe el directorio: {root}")
    docs: list[IngestedDoc] = []
    skipped: list[str] = []
    seen: dict[str, str] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = str(path.relative_to(root))
        if path.name.endswith(".meta.json") or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            skipped.append(rel)
            continue
        doc = ingest_file(path, root=root, config=config)
        if doc.doc_id in seen:
            raise IngestError(f"doc_id repetido {doc.doc_id!r}: {seen[doc.doc_id]} y {rel}")
        seen[doc.doc_id] = rel
        docs.append(doc)
    return docs, skipped
