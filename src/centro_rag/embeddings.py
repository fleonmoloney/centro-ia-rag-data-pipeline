"""Embeddings: tres implementaciones y una guarda de sensibilidad.

* `HashingEmbedder`: determinista y sin red; baseline léxico para tests y evals offline.
  NO es un modelo semántico: no entiende sinónimos.
* `SentenceTransformerEmbedder`: local (extra `local`); los datos no salen del servidor.
* `WatsonxEmbedder`: nube de IBM (extra `watsonx`); solo con `sensitivity="public"`.

Los modelos por defecto son multilingües a propósito: el corpus objetivo está en español y los
modelos "english" (p. ej. slate-125m-english) rinden mal con otro idioma.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import unicodedata
from collections import Counter
from typing import Literal, Protocol

import numpy as np

DEFAULT_LOCAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_WATSONX_MODEL = "ibm/granite-embedding-278m-multilingual"
DEFAULT_WATSONX_URL = "https://us-south.ml.cloud.ibm.com"

Sensitivity = Literal["sensitive", "public"]


class CloudNotAllowedError(PermissionError):
    """Se intentó usar un embedder en la nube con datos marcados como sensibles."""


class Embedder(Protocol):
    name: str
    is_cloud: bool

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


# --------------------------------------------------------------------------- baseline offline
_STOPWORDS = frozenset(
    "de la el los las y en que a un una por con para se del al es lo su sus o como mas ya no si "
    "le les este esta estos estas ese esa entre sobre cada".split()
)


def _tokens(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    words = [w for w in re.findall(r"[a-z0-9]+", folded) if len(w) > 1 and w not in _STOPWORDS]
    return [w[:6] for w in words]  # "stemming" tosco: prefijo de 6 letras


class HashingEmbedder:
    """Bolsa de palabras (+ bigramas) con hashing y signo, normalizada a norma 1."""

    is_cloud = False

    def __init__(self, dim: int = 2048) -> None:
        if dim < 16:
            raise ValueError("dim debe ser >= 16")
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % self.dim, 1.0 if (value >> 63) & 1 else -1.0

    def _embed(self, text: str) -> list[float]:
        toks = _tokens(text)
        counts = Counter(toks)
        counts.update({f"{a}_{b}": 0.5 for a, b in zip(toks, toks[1:], strict=False)})
        vec = np.zeros(self.dim, dtype=np.float64)
        for feature, tf in counts.items():
            idx, sign = self._bucket(feature)
            vec[idx] += sign * (1.0 + math.log(tf)) if tf >= 1 else sign * tf
        norm = np.linalg.norm(vec)
        return (vec / norm).tolist() if norm else vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


# --------------------------------------------------------------------------- local
class SentenceTransformerEmbedder:
    """Modelo local vía sentence-transformers. No se ejecuta en CI (descarga el modelo)."""

    is_cloud = False

    def __init__(
        self,
        model_name: str = DEFAULT_LOCAL_MODEL,
        *,
        query_prefix: str = "",
        passage_prefix: str = "",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depende del entorno
            raise RuntimeError("Falta el extra: pip install '.[local]'") from exc
        self._model = SentenceTransformer(model_name)
        self.name = model_name
        # Modelos tipo e5 requieren prefijos "query: " / "passage: "; MiniLM no.
        self._qp, self._pp = query_prefix, passage_prefix

    def _encode(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode([self._pp + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self._qp + text])[0]


# --------------------------------------------------------------------------- nube
class WatsonxEmbedder:
    """Embeddings de IBM watsonx.ai. Los textos viajan a la nube: solo para datos públicos.

    Variables: WATSONX_API_KEY, WATSONX_PROJECT_ID (obligatorias), WATSONX_URL (opcional).
    Verificá que `model_id` esté disponible en tu región/cuenta; no se prueba contra la API real
    en este repo.
    """

    is_cloud = True

    def __init__(
        self,
        model_id: str = DEFAULT_WATSONX_MODEL,
        *,
        url: str | None = None,
        api_key: str | None = None,
        project_id: str | None = None,
    ) -> None:
        api_key = api_key or os.environ.get("WATSONX_API_KEY")
        project_id = project_id or os.environ.get("WATSONX_PROJECT_ID")
        if not api_key or not project_id:
            raise RuntimeError("Faltan WATSONX_API_KEY y/o WATSONX_PROJECT_ID")
        try:
            from ibm_watsonx_ai import Credentials
            from ibm_watsonx_ai.foundation_models import Embeddings
        except ImportError as exc:  # pragma: no cover - depende del entorno
            raise RuntimeError("Falta el extra: pip install '.[watsonx]'") from exc
        credentials = Credentials(
            url=url or os.environ.get("WATSONX_URL", DEFAULT_WATSONX_URL), api_key=api_key
        )
        self._emb = Embeddings(model_id=model_id, credentials=credentials, project_id=project_id)
        self.name = f"watsonx:{model_id}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._emb.embed_documents(texts=texts)

    def embed_query(self, text: str) -> list[float]:
        return self._emb.embed_query(text=text)


def build_embedder(name: str = "local", *, sensitivity: Sensitivity = "sensitive") -> Embedder:
    """Fábrica con guarda: la nube exige declarar explícitamente `sensitivity="public"`."""
    if name == "hashing":
        return HashingEmbedder()
    if name == "local":
        return SentenceTransformerEmbedder()
    if name == "watsonx":
        if sensitivity != "public":
            raise CloudNotAllowedError(
                "Watsonx envía el texto a la nube de IBM. Los datos están marcados como "
                "'sensitive'; usá un embedder local o declará sensitivity='public' si "
                "corresponde."
            )
        return WatsonxEmbedder()
    raise ValueError(f"Embedder desconocido: {name!r} (hashing, local, watsonx)")
