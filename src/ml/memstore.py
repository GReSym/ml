"""In-memory game vectors with a configurable fusion of the description and review components (offline evaluation only).

Milvus keeps `desc_embedding` and `reviews_embedding` next to the combined `embedding`, so any other way of merging them
can be evaluated without the model: load both components once, build the combined matrix for a fusion spec, and
search it exactly (brute-force cosine, which also removes the HNSW approximation from the comparison).

`InMemoryVectorStore` has the part of `GameVectorStore`'s interface the evaluation uses (`get_vectors`, `search`,
`search_many`), so `evaluation.evaluate` runs on it unchanged.

Fusion specs are `<kind>:<alpha>` (ml.fusion: sum, rcenter, concat, rcenter-concat; alpha = reviews.weight_max). Here the mean
review vector of the centered kinds is computed from the loaded components, not read from reviews.mean_path.
"""

import logging
import time
from pathlib import Path

import numpy as np

from ml.fusion import FusionSpec, unit_rows
from ml.vectors import DESC_VECTOR_FIELD, REVIEWS_VECTOR_FIELD, GameVectorStore

log = logging.getLogger(__name__)

# Query rows multiplied with the whole catalogue at once (30 × 166k floats = 20 MB)
SEARCH_BATCH = 64
ANISOTROPY_SAMPLE = 2000


class GameComponents:
    """Every game's description and review vectors (float32, rows sorted by id) and its number of selected reviews."""

    def __init__(self, ids: np.ndarray, desc: np.ndarray, reviews: np.ndarray, review_counts: np.ndarray) -> None:
        order = np.argsort(ids)
        self.ids = ids[order]
        self.desc = desc[order]
        self.reviews = reviews[order]
        self.review_counts = review_counts[order]
        self.has_reviews = self.review_counts > 0

    @classmethod
    def load(cls, store: GameVectorStore, cache: Path | None = None, refresh: bool = False) -> GameComponents:
        """From the cache file when it exists and holds as many games as the collection (unless `refresh`), else from Milvus."""
        if cache is not None and cache.exists() and not refresh:
            with np.load(cache) as data:
                if int(data["ids"].shape[0]) == store.count():
                    log.info("Vector components from %s (written %s; --refresh-vectors re-reads Milvus)",
                             cache, time.strftime("%Y-%m-%d %H:%M", time.localtime(cache.stat().st_mtime)))
                    return cls(data["ids"], data["desc"], data["reviews"], data["review_counts"])
                log.info("%s has a different number of games than %s, re-reading Milvus", cache, store.collection)
        started = time.monotonic()
        ids: list[np.ndarray] = []
        desc: list[np.ndarray] = []
        reviews: list[np.ndarray] = []
        counts: list[np.ndarray] = []
        for rows in store.raw_rows(["id", DESC_VECTOR_FIELD, REVIEWS_VECTOR_FIELD, "review_ids"], batch_size=1000):
            dim = len(rows[0][DESC_VECTOR_FIELD])
            ids.append(np.asarray([row["id"] for row in rows], dtype=np.int64))
            desc.append(np.asarray([row[DESC_VECTOR_FIELD] for row in rows], dtype=np.float32).reshape(len(rows), dim))
            batch = np.zeros((len(rows), dim), dtype=np.float32)
            batch_counts = np.zeros(len(rows), dtype=np.int16)
            for i, row in enumerate(rows):
                vector = row.get(REVIEWS_VECTOR_FIELD)
                if vector is not None and len(vector) > 0:
                    batch[i] = vector
                    batch_counts[i] = len(row.get("review_ids") or []) or 1
            reviews.append(batch)
            counts.append(batch_counts)
        components = cls(np.concatenate(ids), np.concatenate(desc), np.concatenate(reviews), np.concatenate(counts))
        log.info("Loaded %d games from %s in %.0f s (%d with reviews)", len(components.ids), store.collection,
                 time.monotonic() - started, int(components.has_reviews.sum()))
        if cache is not None:
            tmp = cache.with_suffix(".tmp.npz")
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez(tmp, ids=components.ids, desc=components.desc, reviews=components.reviews,
                         review_counts=components.review_counts)
                tmp.replace(cache)
            except OSError as e:
                log.warning("Couldn't write the cache %s: %s", cache, e)
        return components

    def review_mean(self) -> np.ndarray:
        return self.reviews[self.has_reviews].mean(axis=0)

    def anisotropy(self, seed: int = 0) -> dict[str, float]:
        """Mean cosine between random pairs of different games: description–description, review–review and
        description–review (the cross term of `sum`), plus review–review after centering. High = a shared direction."""
        rng = np.random.default_rng(seed)
        with_reviews = np.flatnonzero(self.has_reviews)
        if len(with_reviews) < 2:
            return {}
        a = rng.choice(with_reviews, ANISOTROPY_SAMPLE)
        b = rng.choice(with_reviews, ANISOTROPY_SAMPLE)
        keep = a != b
        a, b = a[keep], b[keep]
        centered = unit_rows(self.reviews - self.review_mean())

        def mean_cos(x: np.ndarray, y: np.ndarray) -> float:
            return float(np.einsum("ij,ij->i", x, y).mean())

        return {
            "descDesc": mean_cos(self.desc[a], self.desc[b]),
            "reviewsReviews": mean_cos(self.reviews[a], self.reviews[b]),
            "descReviews": mean_cos(self.desc[a], self.reviews[b]),
            "ownDescReviews": mean_cos(self.desc[a], self.reviews[a]),
            "reviewsReviewsCentered": mean_cos(centered[a], centered[b]),
        }

    def fuse(self, spec: FusionSpec, max_selected: int) -> np.ndarray:
        """The search vectors of every game (rows in `ids` order), exactly as ml.vectorize builds them."""
        return spec.fuse(self.desc, self.reviews, self.review_counts, max_selected,
                         self.review_mean() if spec.centered else None)


class InMemoryVectorStore:
    """Exact cosine search over one fused matrix. Duck-types the read side of GameVectorStore used by the evaluation."""

    def __init__(self, ids: np.ndarray, matrix: np.ndarray, collection: str) -> None:
        self.ids = ids
        self.matrix = matrix
        self.collection = collection

    def _rows(self, ids: list[int] | np.ndarray) -> np.ndarray:
        """Row indices of the ids that exist (unknown ids are dropped)."""
        ids = np.asarray(ids, dtype=np.int64)
        if ids.size == 0:
            return np.empty(0, dtype=np.int64)
        rows = np.minimum(np.searchsorted(self.ids, ids), len(self.ids) - 1)
        return rows[self.ids[rows] == ids]

    def get_vectors(self, ids: list[int], batch_size: int = 0) -> dict[int, np.ndarray]:
        rows = self._rows(list(ids))
        return {int(self.ids[row]): self.matrix[row] for row in rows}

    def ids_with_vectors(self, ids: list[int]) -> set[int]:
        return {int(self.ids[row]) for row in self._rows(list(ids))}

    def search(self, vector: list[float], limit: int, exclude_ids: list[int]) -> list[tuple[int, float]]:
        return self.search_many([vector], limit, exclude_ids)[0]

    def search_many(self, vectors: list[list[float]], limit: int, exclude_ids: list[int]) -> list[list[tuple[int, float]]]:
        if not vectors:
            return []
        queries = unit_rows(np.asarray(vectors, dtype=np.float32).reshape(len(vectors), -1))
        excluded = self._rows(exclude_ids)
        limit = min(limit, len(self.ids) - len(excluded))
        result: list[list[tuple[int, float]]] = []
        for start in range(0, len(queries), SEARCH_BATCH):
            scores = queries[start:start + SEARCH_BATCH] @ self.matrix.T
            scores[:, excluded] = -np.inf
            top = np.argpartition(-scores, limit - 1, axis=1)[:, :limit]
            for row, candidates in zip(scores, top):
                order = candidates[np.argsort(-row[candidates], kind="stable")]
                result.append([(int(self.ids[i]), float(row[i])) for i in order])
        return result

    def count(self) -> int:
        return len(self.ids)

    def close(self) -> None:
        pass
