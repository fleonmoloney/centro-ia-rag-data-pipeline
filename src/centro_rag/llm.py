"""Contrato mínimo de LLM (mismo contrato que `centro-ia-agentic-harness`).

Este repo no incluye adaptadores a proveedores: la recuperación es lo que se evalúa aquí.
`ScriptedLLM` permite probar self-query, multi-query y QA de forma determinista.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLM(Protocol):
    def generate(self, prompt: str) -> str: ...

    def structured(self, prompt: str, schema: type[T]) -> T: ...


class ScriptedLLM:
    """LLM de guion: respuestas predefinidas en orden; la última se repite."""

    def __init__(self, texts: list[str] | None = None, structured: list[dict] | None = None):
        self._texts = list(texts or [])
        self._structured = list(structured or [])
        self.prompts: list[str] = []

    @staticmethod
    def _next(queue: list, what: str):
        if not queue:
            raise RuntimeError(f"ScriptedLLM: no hay respuestas de tipo '{what}' en el guion")
        return queue.pop(0) if len(queue) > 1 else queue[0]

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return str(self._next(self._texts, "texto"))

    def structured(self, prompt: str, schema: type[T]) -> T:
        self.prompts.append(prompt)
        return schema.model_validate(self._next(self._structured, "estructurada"))


class StructuredOutputError(RuntimeError):
    """El modelo no produjo una salida válida dentro del límite de intentos."""


def structured_with_retry(
    llm: LLM, prompt: str, schema: type[T], *, max_attempts: int = 3
) -> T:
    """Pide salida estructurada; si no valida, reintenta mostrando el error al modelo."""
    current, last = prompt, ""
    for _ in range(max_attempts):
        try:
            return llm.structured(current, schema)
        except ValidationError as exc:
            last = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            current = f"{prompt}\n\nTu respuesta anterior no validó: {last}\nCorregila."
    raise StructuredOutputError(f"Sin salida válida tras {max_attempts} intentos: {last}")
