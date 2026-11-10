FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ANONYMIZED_TELEMETRY=False

WORKDIR /app

# Núcleo + Chroma. Sin extras pesados (sentence-transformers arrastra torch): agregalos si los usás.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install ".[chroma]"

COPY data/ ./data/
COPY evals/ ./evals/

RUN useradd --create-home --uid 10001 rag
USER rag

# Corre los evals con el corpus sintético; sale con código distinto de 0 si algo falla.
CMD ["python", "-m", "centro_rag.evals"]
