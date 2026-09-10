FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN git init
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --no-install-project

COPY src/ src/
COPY param_config.yaml dvc.yaml dvc.lock ./
COPY .dvc/config .dvc/config

RUN uv sync --frozen

ENTRYPOINT ["sh", "-c", "uv run dvc pull && uv run dvc repro"]

