FROM python:3.10-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --no-install-project

COPY src/ src/
COPY api/ api/
COPY param_config.yaml ./

RUN uv sync --frozen

EXPOSE 8080


CMD ["uv", "run", "fastapi", "run", "api/app.py", "--host", "0.0.0.0", "--port", "8080"]