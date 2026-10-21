# ML service image: one image for `ml api` and `ml worker`. Built by ../deploy/compose.yaml (profile "app").
# The model isn't baked in: compose mounts ml/Qwen3-Embedding at /models/Qwen3-Embedding (read-only).
# Configuration: config.example.json + GRESYM_ML__<SECTION>__<KEY> environment variables (see src/ml/config.py).

FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    HF_HUB_OFFLINE=1 \
    HF_HOME=/tmp/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first (torch is pinned to the pytorch-cpu index in pyproject.toml), cached until uv.lock changes
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project

# Editable install of the project itself: config.py finds config.example.json relative to src/
COPY pyproject.toml uv.lock README.md config.example.json ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

RUN useradd --system --uid 10001 --no-create-home ml
USER ml
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["ml", "api"]
