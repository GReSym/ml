"""Logical backup of the game vectors: Milvus collection ↔ compressed .npz file.

The file holds every field of every row (the three vectors included), so a restore needs no model and no re-embedding.
It doesn't depend on the Milvus version or its on-disk format, so it survives Milvus upgrades and moves between machines.
Restore recreates the collection with the current schema (GameVectorStore.ensure_collection) and upserts the rows as-is.

Layout: one array per field, review_ids flattened with offsets, reviews_embedding with a has_reviews mask
(zeros where the game has no reviews), and a JSON "meta" entry (collection, model, dimension, TEXT_VERSION, ...).
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from ml.config import Settings
from ml.text import TEXT_VERSION
from ml.vectors import DESC_VECTOR_FIELD, REVIEWS_VECTOR_FIELD, VECTOR_FIELD, GameVectorStore

log = logging.getLogger(__name__)

FORMAT_VERSION = 1
BATCH_SIZE = 500
SCALAR_FIELDS = {  # field → numpy dtype
    "id": np.int64,
    "reviews_weight": np.float32,
    "text_version": np.int32,
    "updated_at": np.int64,
    "steam_reviews_total": np.int64,
    "steam_positive_share": np.float32,
}
ALL_FIELDS = ["id", VECTOR_FIELD, DESC_VECTOR_FIELD, REVIEWS_VECTOR_FIELD, "reviews_weight", "text_version",
              "updated_at", "desc_hash", "review_ids", "steam_reviews_total", "steam_positive_share"]


class BackupError(Exception):
    pass


def _store(settings: Settings, collection: str | None) -> GameVectorStore:
    milvus = settings.milvus if collection is None else settings.milvus.model_copy(update={"collection": collection})
    return GameVectorStore(milvus, settings.model)


def export_vectors(settings: Settings, out: Path | None, collection: str | None = None) -> Path:
    store = _store(settings, collection)
    try:
        if not store.exists():
            raise BackupError(f"Collection {store.collection} doesn't exist")
        out = out or Path(f"milvus-{store.collection}-{datetime.now():%Y%m%dT%H%M}.npz")
        if out.exists():
            raise BackupError(f"{out} already exists")

        dim = settings.model.dimension
        columns: dict[str, list] = {name: [] for name in ALL_FIELDS}
        for rows in store.raw_rows(ALL_FIELDS, BATCH_SIZE):
            for row in rows:
                for name in ALL_FIELDS:
                    columns[name].append(row.get(name))
            log.info("Read %d rows", len(columns["id"]))

        count = len(columns["id"])
        has_reviews = np.array([v is not None and len(v) > 0 for v in columns[REVIEWS_VECTOR_FIELD]], dtype=bool)
        zeros = np.zeros(dim, dtype=np.float32)
        review_ids = [list(v or []) for v in columns["review_ids"]]
        arrays: dict[str, np.ndarray] = {
            name: np.asarray(columns[name], dtype=dtype) for name, dtype in SCALAR_FIELDS.items()
        }
        for name in (VECTOR_FIELD, DESC_VECTOR_FIELD):
            arrays[name] = np.asarray(columns[name], dtype=np.float32).reshape(count, dim)
        arrays[REVIEWS_VECTOR_FIELD] = np.asarray(
            [v if ok else zeros for v, ok in zip(columns[REVIEWS_VECTOR_FIELD], has_reviews)], dtype=np.float32
        ).reshape(count, dim)
        arrays["has_reviews"] = has_reviews
        arrays["desc_hash"] = np.asarray(columns["desc_hash"], dtype="<U32")
        arrays["review_ids"] = np.asarray([i for ids in review_ids for i in ids], dtype=np.int64)
        arrays["review_ids_offsets"] = np.cumsum([0, *map(len, review_ids)], dtype=np.int64)
        meta = {
            "format_version": FORMAT_VERSION,
            "collection": store.collection,
            "model": settings.model.name,
            "dimension": dim,
            "text_version": TEXT_VERSION,
            "rows": count,
            "exported_at": int(time.time()),
        }
        arrays["meta"] = np.array(json.dumps(meta))

        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, out)
        log.info("Exported %d vectors from %s → %s (%.1f MB)", count, store.collection, out, out.stat().st_size / 2**20)
        return out
    finally:
        store.close()


def import_vectors(settings: Settings, source: Path, collection: str | None = None, overwrite: bool = False) -> int:
    if not source.exists():
        raise BackupError(f"{source} not found")
    with np.load(source, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        if meta.get("format_version") != FORMAT_VERSION:
            raise BackupError(f"Unsupported backup format {meta.get('format_version')}")
        if meta["dimension"] != settings.model.dimension or meta["model"] != settings.model.name:
            raise BackupError(
                f"Backup is for {meta['model']} (dim {meta['dimension']}), the config uses "
                f"{settings.model.name} (dim {settings.model.dimension})"
            )
        if meta["text_version"] != TEXT_VERSION:
            log.warning("Backup was made with TEXT_VERSION %s, current is %s: `ml stale` will list these vectors "
                        "as outdated", meta["text_version"], TEXT_VERSION)
        arrays = {name: data[name] for name in data.files if name != "meta"}

    store = _store(settings, collection)
    try:
        store.ensure_collection()
        existing = store.count()
        if existing and not overwrite:
            raise BackupError(f"Collection {store.collection} already has {existing} vectors. "
                              "Rows with the same id are replaced, others are kept. Pass --yes to continue")

        count = int(meta["rows"])
        offsets = arrays["review_ids_offsets"]
        for start in range(0, count, BATCH_SIZE):
            rows = []
            for i in range(start, min(start + BATCH_SIZE, count)):
                row = {name: arrays[name][i].item() for name in SCALAR_FIELDS}
                row[VECTOR_FIELD] = arrays[VECTOR_FIELD][i].tolist()
                row[DESC_VECTOR_FIELD] = arrays[DESC_VECTOR_FIELD][i].tolist()
                row[REVIEWS_VECTOR_FIELD] = arrays[REVIEWS_VECTOR_FIELD][i].tolist() if arrays["has_reviews"][i] else None
                row["desc_hash"] = str(arrays["desc_hash"][i])
                row["review_ids"] = arrays["review_ids"][offsets[i]:offsets[i + 1]].tolist()
                rows.append(row)
            store.upsert_rows(rows)
            log.info("Imported %d / %d", start + len(rows), count)
        log.info("Imported %d vectors from %s into %s (collection now has %d)",
                 count, source, store.collection, store.count())
        return count
    finally:
        store.close()
