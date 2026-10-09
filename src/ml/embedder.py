"""Qwen3-Embedding via Sentence Transformers, on CPU."""

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

        log.info("Loading %s from %s", settings.name, path)
        # The checkpoint is bfloat16. Without AVX512-BF16 (e.g. Ryzen 5 5500U) bf16 matmuls on CPU
        # are ~4x slower than fp32, so always load in float32.
        self._model = SentenceTransformer(
            str(path), device="cpu", model_kwargs={"dtype": torch.float32}
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
