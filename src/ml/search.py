"""Game search: weighted reciprocal rank fusion (RRF) of ranked lists.

- Exact search (default): BM25 over titles and descriptions (ml.text_index). A title hit has its own list with a
  higher weight, so "Satisfactory" ranks the game itself above the games that mention it. No model.
- Description search ("игра-песочница с открытым миром"): the same BM25 lists plus the nearest game vectors to the
  embedded query. The model is loaded on the first such request (~2.4 GB RAM in fp32, ~10–20 s), never before.

RRF only uses ranks, so BM25 scores and cosine similarities don't need to be on one scale:
score(game) = Σ weight_list / (k + rank_in_list), rank from 1.
"""

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from ml.config import Settings
from ml.embedder import Embedder, create_embedder
from ml.text_index import TEXT_FIELD, TITLE_FIELD, GameTextStore
from ml.vectors import GameVectorStore

log = logging.getLogger(__name__)

SearchMode = Literal["bm25", "hybrid"]


@dataclass(frozen=True)
class SearchResult:
    items: list[tuple[int, float]]  # (game_id, fused score), best first
    mode: SearchMode  # "bm25" also when a description search fell back (dense disabled or model unavailable)


def rrf(lists: list[tuple[list[int], float]], k: int) -> list[tuple[int, float]]:
    """Weighted RRF of ranked id lists (best first), best fused score first; ties broken by the lower id."""
    scores: dict[int, float] = defaultdict(float)
    for ids, weight in lists:
        for rank, game_id in enumerate(ids, start=1):
            scores[game_id] += weight / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class GameSearch:
    """Thread-safe: the embedder is created once, behind a lock."""

    def __init__(self, settings: Settings, texts: GameTextStore, vectors: GameVectorStore) -> None:
        self._settings = settings
        self._texts = texts
        self._vectors = vectors
        self._embedder: Embedder | None = None
        self._embedder_failed = False
        self._lock = threading.Lock()

    @property
    def model_loaded(self) -> bool:
        return self._embedder is not None

    def _get_embedder(self) -> Embedder | None:
        if self._embedder is not None or self._embedder_failed:
            return self._embedder
        with self._lock:
            if self._embedder is None and not self._embedder_failed:
                try:
                    self._embedder = create_embedder(self._settings.model)
                except Exception:
                    # Not retried until restart: loading takes seconds and would fail the same way on every request
                    log.exception("Could not load the embedding model, description search falls back to BM25")
                    self._embedder_failed = True
        return self._embedder

    def search(self, query: str, exact: bool, limit: int) -> SearchResult:
        """Blocking (Milvus, embedding): call it from a thread."""
        if not query.strip():
            return SearchResult(items=[], mode="bm25")
        settings = self._settings.search
        n = max(limit, settings.candidates)
        keyword = self._texts.search(query, n)
        lists = [(keyword[TITLE_FIELD], settings.title_weight), (keyword[TEXT_FIELD], settings.text_weight)]

        mode: SearchMode = "bm25"
        if not exact and settings.dense:
            embedder = self._get_embedder()
            if embedder is not None:
                hits = self._vectors.search(embedder.embed_query(query), n, exclude_ids=[])
                lists.append(([game_id for game_id, _ in hits], settings.dense_weight))
                mode = "hybrid"

        return SearchResult(items=rrf(lists, settings.rrf_k)[:limit], mode=mode)
