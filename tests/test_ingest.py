import json

import pytest
from pdfs import make_blank_pdf, make_pdf

from centro_rag.ingest import (
    ChunkConfig,
    IngestError,
    NoTextError,
    ingest_directory,
    ingest_file,
    parse_front_matter,
    slugify,
)

MD = """---
year: 2024
institution: Hospital Modelo
document_type: protocolo
---
# Título
## Sección A
Texto de la sección A sobre calibración.
## Sección B
Texto de la sección B sobre registro.
"""


def write(tmp_path, name, content):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_slugify_removes_accents_and_symbols():
    assert slugify("Protocolo Tomógrafo (2024).md") == "protocolo-tomografo-2024-md"
    assert slugify("###") == "doc"


def test_front_matter_parses_year_as_int():
    meta, body = parse_front_matter(MD)
    assert meta == {"year": 2024, "institution": "Hospital Modelo", "document_type": "protocolo"}
    assert body.startswith("# Título")


def test_front_matter_rejects_bad_year_and_bad_lines():
    with pytest.raises(IngestError):
        parse_front_matter("---\nyear: dos mil\n---\ntexto")
    with pytest.raises(IngestError):
        parse_front_matter("---\nesto no tiene dos puntos\n---\ntexto")


def test_text_without_front_matter_is_untouched():
    assert parse_front_matter("hola") == ({}, "hola")


def test_chunks_carry_metadata_and_stable_ids(tmp_path):
    doc = ingest_file(write(tmp_path, "Protocolo Uno.md", MD), root=tmp_path)
    assert doc.doc_id == "protocolo-uno"
    first = doc.chunks[0]
    assert first.id == "protocolo-uno:0"
    for key in ("doc_id", "section_id", "chunk_index", "source", "content_hash", "year"):
        assert key in first.metadata
    assert first.metadata["institution"] == "Hospital Modelo"
    assert first.metadata["section_id"] == "protocolo-uno#s0"


def test_header_path_is_used_for_embedding_but_not_stored_text(tmp_path):
    doc = ingest_file(write(tmp_path, "a.md", MD))
    chunk = next(c for c in doc.chunks if "sección A" in c.text)
    assert chunk.context == "Título > Sección A"
    assert chunk.embed_text.startswith("Título > Sección A\n")
    assert not chunk.text.startswith("Título > Sección A\n")


def test_parents_include_document_and_each_section(tmp_path):
    doc = ingest_file(write(tmp_path, "a.md", MD))
    keys = [p.key for p in doc.parents]
    assert keys[0] == "a" and doc.parents[0].level == "document"
    assert len([p for p in doc.parents if p.level == "section"]) >= 2
    full = doc.parents[0].text
    assert "sección A" in full and "sección B" in full


def test_content_hash_is_stable_and_sensitive(tmp_path):
    path = write(tmp_path, "a.md", MD)
    h1 = ingest_file(path).content_hash
    assert ingest_file(path).content_hash == h1
    assert ingest_file(path, config=ChunkConfig(chunk_size=300)).content_hash != h1
    write(tmp_path, "a.md", MD.replace("calibración", "mantenimiento"))
    assert ingest_file(path).content_hash != h1
    assert ingest_file(path, metadata={"year": 1999}).content_hash != h1


def test_sidecar_metadata_for_files_without_front_matter(tmp_path):
    path = write(tmp_path, "nota.txt", "Texto plano sin front matter.")
    (tmp_path / "nota.txt.meta.json").write_text(
        json.dumps({"year": "2021", "institution": "Instituto Modelo"}), encoding="utf-8"
    )
    doc = ingest_file(path)
    assert doc.metadata == {"year": 2021, "institution": "Instituto Modelo"}


def test_explicit_metadata_overrides_front_matter(tmp_path):
    doc = ingest_file(write(tmp_path, "a.md", MD), metadata={"year": 2030})
    assert doc.metadata["year"] == 2030


def test_pdf_with_text_is_ingested_with_page_number(tmp_path):
    path = tmp_path / "doc.pdf"
    make_pdf("Protocolo de ejemplo sobre calibracion", path)
    doc = ingest_file(path)
    assert doc.chunks and doc.chunks[0].metadata["page"] == 1
    assert "calibracion" in doc.parents[0].text


def test_scanned_pdf_fails_loudly_instead_of_indexing_nothing(tmp_path):
    path = tmp_path / "escaneado.pdf"
    make_blank_pdf(path)
    with pytest.raises(NoTextError, match="OCR"):
        ingest_file(path)


def test_corrupt_pdf_is_reported(tmp_path):
    path = tmp_path / "roto.pdf"
    path.write_bytes(b"esto no es un pdf")
    with pytest.raises(IngestError):
        ingest_file(path)


def test_csv_and_json_are_supported(tmp_path):
    csv_doc = ingest_file(write(tmp_path, "t.csv", "equipo,estado\ntomografo,activo\n"))
    assert "equipo: tomografo | estado: activo" in csv_doc.parents[0].text
    json_doc = ingest_file(write(tmp_path, "d.json", '{"a": "ñandú"}'))
    assert "ñandú" in json_doc.parents[0].text
    with pytest.raises(IngestError):
        ingest_file(write(tmp_path, "mal.json", "{no es json"))


def test_unsupported_missing_and_non_utf8(tmp_path):
    with pytest.raises(IngestError):
        ingest_file(write(tmp_path, "a.xyz", "x"))
    with pytest.raises(IngestError):
        ingest_file(tmp_path / "no-existe.md")
    bad = tmp_path / "latin.txt"
    bad.write_bytes("acción".encode("latin-1"))
    with pytest.raises(IngestError, match="UTF-8"):
        ingest_file(bad)


def test_empty_document_is_an_error(tmp_path):
    with pytest.raises(NoTextError):
        ingest_file(write(tmp_path, "vacio.txt", "   \n"))


def test_directory_skips_unsupported_and_sidecars(tmp_path):
    write(tmp_path, "a.md", MD)
    write(tmp_path, "sub/b.txt", "Texto de b.")
    write(tmp_path, "foto.png", "x")
    (tmp_path / "b.txt.meta.json").write_text("{}", encoding="utf-8")
    docs, skipped = ingest_directory(tmp_path)
    assert sorted(d.doc_id for d in docs) == ["a", "sub-b"]
    assert sorted(skipped) == ["b.txt.meta.json", "foto.png"]


def test_directory_detects_colliding_doc_ids(tmp_path):
    write(tmp_path, "a-b.md", MD)
    write(tmp_path, "a/b.md", MD)
    with pytest.raises(IngestError, match="repetido"):
        ingest_directory(tmp_path)


def test_chunk_config_validation():
    with pytest.raises(ValueError):
        ChunkConfig(chunk_size=10)
    with pytest.raises(ValueError):
        ChunkConfig(chunk_size=100, chunk_overlap=100)


def test_long_text_is_split_within_the_limit(tmp_path):
    body = "\n\n".join(f"Párrafo {i} " + "palabra " * 40 for i in range(12))
    doc = ingest_file(write(tmp_path, "largo.txt", body), config=ChunkConfig(300, 40))
    assert len(doc.chunks) > 5
    assert max(len(c.text) for c in doc.chunks) <= 300
    assert [c.metadata["chunk_index"] for c in doc.chunks] == list(range(len(doc.chunks)))
