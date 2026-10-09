"""Game text → embedding → Milvus. Used by the queue worker and the bulk CLI command."""

import logging
import time
from dataclasses import dataclass

from ml.config import Settings
from ml.db import GameRepository
from ml.embedder import Embedder
from ml.text import TEXT_VERSION, build_game_text
from ml.vectors import GameVectorStore

log = logging.getLogger(__name__)


@dataclass
class VectorizeResult:
    upserted: list[int]
    deleted: list[int]


class Vectorizer:
    """Loads the model (~2.4 GB RAM in fp32). Not thread-safe, call from one thread."""

    def __init__(self, settings: Settings) -> None:
        self.games = GameRepository(settings.database)
        self.store = GameVectorStore(settings.milvus, settings.model)
        self.store.ensure_collection()
        self._embedder = Embedder(settings.model)

    def vectorize(self, ids: list[int]) -> VectorizeResult:
        """Idempotent: re-vectorizing a game overwrites its vector. Games missing in MariaDB are removed from Milvus."""
        games = self.games.get_games(ids)
        found = {game.id for game in games}
        missing = [game_id for game_id in ids if game_id not in found]

        if games:
            texts = [build_game_text(game) for game in games]
            embeddings = self._embedder.embed_documents(texts)
            self.store.upsert({game.id: vector for game, vector in zip(games, embeddings)}, TEXT_VERSION)
        self.store.delete(missing)

        return VectorizeResult(upserted=sorted(found), deleted=missing)

    def close(self) -> None:
        self.games.close()
        self.store.close()


def run_bulk(settings: Settings, ids: list[int] | None, limit: int | None, skip_existing: bool, chunk: int = 32) -> None:
    """Vectorizes the given ids, or the first `limit` games by id (all games if limit is None).

    Works in chunks and upserts after each one, so an interrupted run keeps its progress
    (re-run with skip_existing to continue).
    """
    vectorizer = Vectorizer(settings)
    try:
        if ids is None:
            ids = vectorizer.games.get_game_ids(limit=limit)
        log.info("Vectorizing %d games into %s", len(ids), vectorizer.store.collection)

        started = time.monotonic()
        done = skipped = 0
        for start in range(0, len(ids), chunk):
            batch = ids[start : start + chunk]
            if skip_existing:
                existing = vectorizer.store.existing_ids(batch, TEXT_VERSION)
                skipped += len(existing)
                batch = [game_id for game_id in batch if game_id not in existing]
            if batch:
                result = vectorizer.vectorize(batch)
                done += len(result.upserted)
                if result.deleted:
                    log.warning("Not found in MariaDB (removed from Milvus): %s", result.deleted)

            processed = min(start + chunk, len(ids))
            elapsed = time.monotonic() - started
            rate = done / elapsed if elapsed else 0.0
            eta = (len(ids) - processed) / rate if rate else 0.0
            log.info("%d/%d processed (%d vectorized, %d skipped), %.2f games/s, ETA %.0f min",
                     processed, len(ids), done, skipped, rate, eta / 60)

        log.info("Done in %.1f s. Collection now has %d vectors", time.monotonic() - started, vectorizer.store.count())
    finally:
        vectorizer.close()
