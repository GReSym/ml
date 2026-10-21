"""Qwen3-Embedding via Sentence Transformers, on CPU (default) or a GPU (CUDA/ROCm, model.device)."""

import logging

import torch
from sentence_transformers import SentenceTransformer

from ml.config import ModelSettings

log = logging.getLogger(__name__)


class Embedder:
    def __init__(self, settings: ModelSettings) -> None:
        if settings.threads:
            torch.set_num_threads(settings.threads)

        path = settings.resolved_path
        if not path.exists():
            raise FileNotFoundError(f"Model not found at {path} (model.path in config)")

        device = settings.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"model.device={settings.device}, but torch {torch.__version__} sees no GPU "
                "(CPU build of torch, or the container has no GPU access)"
            )

        log.info("Loading %s from %s on %s (%s, torch %s)", settings.name, path, device, settings.dtype, torch.__version__)
        # The checkpoint is bfloat16. Without AVX512-BF16 (e.g. Ryzen 5 5500U) bf16 matmuls on CPU
        # are ~4x slower than fp32, so the default is float32. On a GPU float16/bfloat16 is faster.
        self._model = SentenceTransformer(
            str(path), device=device, model_kwargs={"dtype": getattr(torch, settings.dtype)}
        )
        self._model.max_seq_length = settings.max_seq_length
        self._batch_size = settings.batch_size

        dimension = self._model.get_embedding_dimension()
        if dimension != settings.dimension:
            raise ValueError(f"Model outputs {dimension} dims, config says model.dimension={settings.dimension}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Game texts are documents: no instruction prompt. Output is L2-normalized."""
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()
