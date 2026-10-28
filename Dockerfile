# ML service image: one image for `ml api` and `ml worker`. Built by ../deploy/compose.yaml (profile "app").
# The model isn't baked in: compose mounts ml/Qwen3-Embedding and ml/Qwen3-Embedding-onnx under /models (read-only).
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

# Dependencies first (uv.lock pins the CPU torch), cached until uv.lock changes
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project

# libatomic1: the ROCm torch wheel imports libatomic, which python:3.14-slim lacks (worker crashed on `import torch`).
# libnuma1: a runtime dependency of the ROCm stack. It does NOT silence the "E-001h rocSHMEM Could not open libnuma"
# startup warning (verified 2026-10-24; probably wants the unversioned libnuma.so) — the warning is harmless and
# left as is, see docs/known-issues.md.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libatomic1 libnuma1 && \
    rm -rf /var/lib/apt/lists/*

# GPU builds replace the locked CPU torch with a build from TORCH_INDEX_URL (set by deploy/manage.sh setup from
# TORCH_BACKEND; stable indexes https://download.pytorch.org/whl/rocm7.14 / cu132). The wheel is not pinned in uv.lock:
# each torch/ROCm release moves it, so the index in deploy/manage.sh has to be revisited when a newer torch is wanted
# (TORCH_INDEX_URL=<url> ./manage.sh setup overrides it without editing manage.sh).
# The rest of the environment stays as locked; torch's own GPU dependencies (triton, nvidia-*/rocm-*) come from the same index.
# Uninstall first: otherwise uv keeps the installed CPU torch as satisfying "torch". `--index` (not --index-url) gives the
# torch index priority over PyPI, which also has torch (with CUDA) and would win under uv's first-index strategy.
ARG TORCH_BACKEND=cpu
ARG TORCH_INDEX_URL=
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$TORCH_BACKEND" != "cpu" ]; then \
        [ -n "$TORCH_INDEX_URL" ] || { echo "TORCH_INDEX_URL is required for TORCH_BACKEND=$TORCH_BACKEND"; exit 1; }; \
        uv pip uninstall --python /app/.venv/bin/python torch \
        && uv pip install --python /app/.venv/bin/python --prerelease allow --index "$TORCH_INDEX_URL" torch; \
    fi \
    && /app/.venv/bin/python -c "import torch; print('torch', torch.__version__, 'hip', torch.version.hip, 'cuda', torch.version.cuda)"

# The project itself, editable (config.py finds config.example.json relative to src/). `uv pip` instead of
# `uv sync`, so a swapped GPU torch isn't reverted to the locked CPU one
COPY pyproject.toml uv.lock README.md config.example.json ./
COPY src ./src
# chmod: copied files keep their host mode, but the container runs as the non-root "ml" user
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /app/.venv/bin/python --no-deps -e . \
    && chmod -R a+rX /app/src /app/config.example.json

RUN useradd --system --uid 10001 --no-create-home ml
USER ml
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["ml", "api"]
