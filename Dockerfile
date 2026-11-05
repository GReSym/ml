# ML service image: one image for `ml api` and `ml worker`. Built by ../deploy/compose.yaml (profile "app").
# Thin on purpose: the heavy base — locked environment, apt libs and the ~18 GB GPU torch — is the local
# gresym/ml-torch:<tag> image built from Dockerfile.torch by ../deploy/manage.sh build-torch (auto-built by
# `up --app` when missing). Code changes rebuild this image in seconds, even after `docker builder prune`,
# and never re-download torch. Rebuild the base (./manage.sh build-torch) only when uv.lock or TORCH_VERSION changes.
# The model isn't baked in: compose mounts ml/Qwen3-Embedding and ml/Qwen3-Embedding-onnx under /models (read-only).
# Configuration: config.example.json + GRESYM_ML__<SECTION>__<KEY> environment variables (see src/ml/config.py).

# deploy/.env → TORCH_BASE_TAG (cpu | rocm-<version> | cuda-<version>), written by ../deploy/manage.sh setup
ARG TORCH_BASE
FROM ${TORCH_BASE}

# The base disables bytecode compilation for the huge GPU torch swap (pointless for ~18 GB of native libs);
# re-enable it for the small project install (faster imports at runtime)
ENV UV_COMPILE_BYTECODE=1

WORKDIR /app

# The project itself, editable (config.py finds config.example.json relative to src/). `uv pip` instead of
# `uv sync`, so the GPU torch from the base isn't reverted to the locked CPU one
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
