"""`ml recombine`: rebuild the search vectors of a collection with the configured fusion (ml.fusion), without the model.

Every row of the source collection keeps its description and review components, so a different fusion (or α) only
recomputes `embedding` and `reviews_weight`: the rows are streamed from the source into a target collection created with
the configured fusion's dimension; every other field (snapshot, review ids, updated_at) is copied as-is.

For the rcenter fusions the mean review vector is frozen in `reviews.mean_path`: computed here from the source's review
vectors if the file doesn't exist yet, read from it otherwise (so a rebuild centers exactly like the vectors already
written, and ml.vectorize centers new games the same way).
"""

import logging
import time

import numpy as np

from ml.backup import ALL_FIELDS, BATCH_SIZE
from ml.config import Settings
from ml.fusion import load_review_mean, save_review_mean
from ml.vectors import DESC_VECTOR_FIELD, REVIEWS_VECTOR_FIELD, VECTOR_FIELD, GameVectorStore

log = logging.getLogger(__name__)


class RecombineError(Exception):
    pass


def _reviews(row: dict) -> list[float] | None:
    vector = row.get(REVIEWS_VECTOR_FIELD)
    return list(vector) if vector is not None and len(vector) > 0 else None


def recombine(settings: Settings, source: str, target: str | None = None, overwrite: bool = False) -> int:
    spec = settings.reviews.spec
    target = target or settings.milvus.collection
    if target == source:
        raise RecombineError("Source and target collection are the same: write into a new collection")
    # The source may use another fusion (another dimension): it's only read, never checked or created
    src = GameVectorStore(settings.milvus.model_copy(update={"collection": source}), settings.model)
    if not src.exists():
        raise RecombineError(f"Collection {source} doesn't exist")

    mean = None
    if spec.centered:
        path = settings.reviews.mean_file
        if path is None:
            raise RecombineError(f"reviews.fusion={spec.kind} needs reviews.mean_path")
        if path.exists():
            mean = load_review_mean(path)
            log.info("Using the frozen mean review vector %s", path)
        else:
            total = np.zeros(settings.model.dimension, dtype=np.float64)
            count = 0
            for rows in src.raw_rows(["id", REVIEWS_VECTOR_FIELD], batch_size=BATCH_SIZE):
                for row in rows:
                    if (vector := _reviews(row)) is not None:
                        total += np.asarray(vector, dtype=np.float64)
                        count += 1
            if count == 0:
                raise RecombineError(f"{source} has no review vectors to center with")
            mean = (total / count).astype(np.float32)
            save_review_mean(path, mean)
            log.info("Mean review vector of %d games written to %s", count, path)

    dst = GameVectorStore.from_settings(settings, target)
    if dst.exists() and dst.count() and not overwrite:
        raise RecombineError(f"{target} already has {dst.count()} vectors (--yes to overwrite them)")
    dst.ensure_collection()
    log.info("Recombining %s → %s with fusion %s (search dim %d)", source, target, spec.name, dst.search_dimension)

    started = time.monotonic()
    written = 0
    max_selected = settings.reviews.max_selected
    dim = settings.model.dimension
    for rows in src.raw_rows(ALL_FIELDS, batch_size=BATCH_SIZE):
        desc = np.asarray([row[DESC_VECTOR_FIELD] for row in rows], dtype=np.float32).reshape(len(rows), dim)
        reviews = np.zeros_like(desc)
        counts = np.zeros(len(rows), dtype=np.int64)
        for i, row in enumerate(rows):
            if (vector := _reviews(row)) is not None:
                reviews[i] = vector
                counts[i] = len(row.get("review_ids") or []) or 1
        embeddings = spec.fuse(desc, reviews, counts, max_selected, mean)
        weights = spec.weights(counts, max_selected) * (counts > 0)
        for i, row in enumerate(rows):
            row[VECTOR_FIELD] = embeddings[i].tolist()
            row["reviews_weight"] = float(weights[i])
            row[DESC_VECTOR_FIELD] = desc[i].tolist()
            row[REVIEWS_VECTOR_FIELD] = _reviews(row)
            row["review_ids"] = list(row.get("review_ids") or [])
        dst.upsert_rows(rows)
        written += len(rows)
        if written % 20_000 < BATCH_SIZE:
            log.info("%d rows written (%.0f s)", written, time.monotonic() - started)
    log.info("Done: %d rows in %s in %.0f s. Set milvus.collection to %s to use it", written, target,
             time.monotonic() - started, target)
    src.close()
    dst.close()
    return written
