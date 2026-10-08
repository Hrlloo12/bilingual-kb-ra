FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
RUN pip install torch==2.6.0 --index-url ${TORCH_INDEX_URL}
COPY requirements.txt .
RUN grep -v -E "^(torch|weasyprint|pytest|datasets|accelerate)[=><]" requirements.txt > requirements-serving.txt \
    && pip install -r requirements-serving.txt

COPY pyproject.toml ./
COPY src ./src
COPY demo/web ./demo/web
RUN pip install --no-deps -e .
COPY configs ./configs
COPY scripts ./scripts
COPY data/corpus ./data/corpus
COPY data/facts ./data/facts
COPY data/interactive ./data/interactive
COPY data/train.jsonl data/validation.jsonl data/test.jsonl data/smoke_queries.jsonl ./data/

RUN chmod -R a+rX /app \
    && useradd --create-home --uid 1000 app \
    && mkdir -p /models /app/results \
    && chown -R app /models /app/results /app/data/corpus/processed
USER app

EXPOSE 8080
CMD ["python", "-m", "rag.api"]
