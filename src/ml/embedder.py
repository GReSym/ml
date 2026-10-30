"""Qwen3-Embedding: ONNX Runtime via FastEmbed on CPU (default), or Sentence Transformers on torch (model.backend).

Both backends produce the same vectors (cosine 1.00000 on 278 texts, fp32), so vectors from either can be mixed.
The ONNX backend is ~1.55x faster on CPU with ~30% less RAM; torch is what runs on a GPU (CUDA/ROCm).
"""

import logging
from typing import Protocol

from ml.config import ModelSettings

log = logging.getLogger(__name__)


# Qwen3-Embedding is instruction-aware: queries take a one-sentence task, documents none. FastEmbed doesn't add it
QUERY_INSTRUCTION = "Given a description of a video game a player wants, retrieve games that match it"


def query_prompt(query: str) -> str:
    return f"Instruct: {QUERY_INSTRUCTION}\nQuery:{query}"


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Game texts are documents: no instruction prompt. Output is L2-normalized."""
        ...

    def embed_query(self, query: str) -> list[float]:
        """A search query, with the instruction prompt. Output is L2-normalized."""
        ...


def create_embedder(settings: ModelSettings) -> Embedder:
    path = settings.resolved_path
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path} (model.path / model.onnx_path in config)")
    return OnnxEmbedder(settings) if settings.backend == "onnx" else TorchEmbedder(settings)


def _check_dimension(dimension: int, settings: ModelSettings) -> None:
    if dimension != settings.dimension:
        raise ValueError(f"Model outputs {dimension} dims, config says model.dimension={settings.dimension}")


class OnnxEmbedder:
    """FastEmbed's built-in model `model.name` (last-token pooling + normalization), loaded from `model.onnx_path`."""

    def __init__(self, settings: ModelSettings) -> None:
        if settings.device != "cpu":
            raise ValueError(f"model.backend=onnx runs on CPU only, model.device={settings.device}: use backend torch")

        from fastembed import TextEmbedding

        path = settings.resolved_path
        log.info("Loading %s from %s (ONNX Runtime, threads %s)", settings.name, path, settings.threads or "default")
        self._model = TextEmbedding(settings.name, specific_model_path=str(path), threads=settings.threads)
        # FastEmbed truncates at the tokenizer's model_max_length (32768) and has no option for it
        self._model.model.tokenizer.enable_truncation(max_length=settings.max_seq_length, direction="right")
        self._batch_size = settings.batch_size
        _check_dimension(self._model.embedding_size, settings)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts, batch_size=self._batch_size)]

    def embed_query(self, query: str) -> list[float]:
        return self.embed_documents([query_prompt(query)])[0]


class TorchEmbedder:
    """Sentence Transformers on CPU or a GPU (model.device)."""

    def __init__(self, settings: ModelSettings) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        if settings.threads:
            torch.set_num_threads(settings.threads)

        device = settings.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"model.device={settings.device}, but torch {torch.__version__} sees no GPU "
                "(CPU build of torch, or the container has no GPU access)"
            )

        path = settings.resolved_path
        log.info("Loading %s from %s on %s (%s, torch %s)", settings.name, path, device, settings.dtype, torch.__version__)
        # The checkpoint is bfloat16. Without AVX512-BF16 (e.g. Ryzen 5 5500U) bf16 matmuls on CPU
        # are ~4x slower than fp32, so the default is float32. On a GPU float16/bfloat16 is faster.
        self._model = SentenceTransformer(
            str(path), device=device, model_kwargs={"dtype": getattr(torch, settings.dtype)}
        )
        self._model.max_seq_length = settings.max_seq_length
        self._batch_size = settings.batch_size
        _check_dimension(self._model.get_embedding_dimension(), settings)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.embed_documents([query_prompt(query)])[0]
