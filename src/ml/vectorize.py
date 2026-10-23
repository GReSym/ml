"""Game text → embedding → Milvus. Used by the queue worker and the bulk CLI command."""

import logging
import time
from dataclasses import dataclass

import numpy as np

from ml.config import ReviewSettings, Settings
from ml.db import GameRepository, GameText, ReviewRow
from ml.embedder import create_embedder
from ml.reviews import SelectedReview, positive_share, reviews_weight, select_reviews
from ml.text import TEXT_VERSION, build_game_text, text_hash
from ml.vectors import GameVector, GameVectorStore

log = logging.getLogger(__name__)


@dataclass
class VectorizeResult:
    upserted: list[int]
    deleted: list[int]


@dataclass(frozen=True)
class GameInput:
    """Everything a game's vector is built from. Also used by the staleness check (without embedding)."""
    game: GameText
    text: str
    desc_hash: str
    reviews: list[SelectedReview]
    positive_share: float | None

    @property
    def steam_reviews_total(self) -> int | None:
        if self.game.steam_positive is None or self.game.steam_negative is None:
            return None
        return self.game.steam_positive + self.game.steam_negative


def build_input(game: GameText, reviews: list[ReviewRow], settings: ReviewSettings) -> GameInput:
    text = build_game_text(game)
    share = positive_share(game.steam_positive, game.steam_negative)
    return GameInput(game, text, text_hash(text), select_reviews(reviews, share, settings), share)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def combine(desc: np.ndarray, reviews: np.ndarray | None, weight: float) -> np.ndarray:
    """normalize(v_desc + α·v_reviews). Both inputs are unit vectors."""
    if reviews is None or weight == 0:
        return desc
    return _normalize(desc + weight * reviews)


class Vectorizer:
    """Loads the model (~2.4 GB RAM in fp32). Not thread-safe, call from one thread."""

    def __init__(self, settings: Settings) -> None:
        self._reviews = settings.reviews
        self.games = GameRepository(settings.database)
        self.store = GameVectorStore(settings.milvus, settings.model)
        self.store.ensure_collection()
        self._embedder = create_embedder(settings.model)

    def vectorize(self, ids: list[int]) -> VectorizeResult:
        """Idempotent: re-vectorizing a game overwrites its vector. Games missing in MariaDB are removed from Milvus."""
        games = self.games.get_games(ids)
        found = {game.id for game in games}
        missing = [game_id for game_id in ids if game_id not in found]

        if games:
            reviews = self.games.get_reviews(sorted(found))
            inputs = [build_input(game, reviews.get(game.id, []), self._reviews) for game in games]
            self.store.upsert([self._embed(item) for item in inputs])
        self.store.delete(missing)

        return VectorizeResult(upserted=sorted(found), deleted=missing)

    def _embed(self, item: GameInput) -> GameVector:
        desc = np.asarray(self._embedder.embed_documents([item.text])[0], dtype=np.float32)
        reviews: np.ndarray | None = None
        if item.reviews:
            # Each review is its own document: short sequences are cheap on CPU and no review drowns the others
            embedded = np.asarray(self._embedder.embed_documents([r.text for r in item.reviews]), dtype=np.float32)
            reviews = _normalize(embedded.mean(axis=0))
        weight = reviews_weight(len(item.reviews), self._reviews)
        return GameVector(
            id=item.game.id,
            embedding=combine(desc, reviews, weight).tolist(),
            desc_embedding=desc.tolist(),
            reviews_embedding=None if reviews is None else reviews.tolist(),
            reviews_weight=weight,
            text_version=TEXT_VERSION,
            desc_hash=item.desc_hash,
            review_ids=[r.id for r in item.reviews],
            steam_reviews_total=item.steam_reviews_total,
            steam_positive_share=item.positive_share,
        )

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
