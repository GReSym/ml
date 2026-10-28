"""Catalogue-wide inputs of the recommendation algorithm, kept in memory by the HTTP API.

- the mean game vector (estimated from a seeded random sample; the vectors are centered with it, recommend.center),
- Steam recommendations counts (the popularity term),
- the most popular games with their centered vectors (popular candidates for the popularity term).

Loaded lazily on the first request and reloaded every RELOAD_SECONDS, so new counts and vectors get picked up.
"""

import logging
import random
import threading
import time
from dataclasses import dataclass

import numpy as np

from ml import recommend as rec
from ml.config import Settings
from ml.db import GameRepository
from ml.vectors import GameVectorStore

log = logging.getLogger(__name__)

# Games whose vectors estimate the mean (the evaluation uses the same size)
MEAN_SAMPLE = 5000
# Popular games kept with vectors: popular candidates come from them, minus the games the user rated
POPULAR_LOADED = 3000
RELOAD_SECONDS = 6 * 3600


@dataclass(frozen=True)
class CatalogueStats:
    mean: np.ndarray
    counts: dict[int, int]  # game id → Steam recommendations count, only games that have one
    max_count: int
    popular_ids: list[int]  # most recommended first
    popular_vectors: np.ndarray  # centered, rows = popular_ids

    def popularity(self, ids: list[int]) -> np.ndarray:
        return rec.popularity(np.asarray([self.counts.get(i, 0) for i in ids], dtype=np.float64), self.max_count)


def load_stats(store: GameVectorStore, games: GameRepository, seed: int = 42) -> CatalogueStats:
    started = time.monotonic()
    ids = games.get_game_ids()
    sample = store.get_vectors(random.Random(seed).sample(ids, min(MEAN_SAMPLE, len(ids))))
    if not sample:
        raise ValueError("No game vectors in Milvus")
    mean = np.asarray(list(sample.values()), dtype=np.float32).mean(axis=0)
    popular = games.get_popular_game_ids(POPULAR_LOADED)
    vectors = store.get_vectors(popular)
    popular = [i for i in popular if i in vectors]
    counts = games.get_recommendation_counts()
    log.info("Catalogue stats loaded in %.1f s: mean of %d vectors, %d popular games, %d counts",
             time.monotonic() - started, len(sample), len(popular), len(counts))
    return CatalogueStats(
        mean=mean,
        counts=counts,
        max_count=max(counts.values(), default=1),
        popular_ids=popular,
        popular_vectors=rec.center(np.asarray([vectors[i] for i in popular], dtype=np.float32).reshape(len(popular), -1), mean),
    )


class CatalogueCache:
    """Thread-safe: FastAPI runs sync endpoints in a thread pool."""

    def __init__(self, settings: Settings, store: GameVectorStore) -> None:
        self._settings = settings
        self._store = store
        self._lock = threading.Lock()
        self._stats: CatalogueStats | None = None
        self._loaded_at = 0.0

    def get(self) -> CatalogueStats:
        with self._lock:
            if self._stats is None or time.monotonic() - self._loaded_at > RELOAD_SECONDS:
                games = GameRepository(self._settings.database)  # not thread-safe, so one per load
                try:
                    self._stats = load_stats(self._store, games)
                finally:
                    games.close()
                self._loaded_at = time.monotonic()
            return self._stats
