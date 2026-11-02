"""Logical backup of the game vectors: Milvus collection ↔ compressed .npz file.

The file holds every field of every row (the three vectors included), so a restore needs no model and no re-embedding.
It doesn't depend on the Milvus version or its on-disk format, so it survives Milvus upgrades and moves between machines.
Restore recreates the collection with the current schema (GameVectorStore.ensure_collection) and upserts the rows as-is.

The rows are split into chunks (CHUNK_SIZE rows each), written to the archive one at a time and read back one at a time,
so neither side ever holds more than one chunk in memory (a whole catalogue as Python lists is ~16 GB).
Layout: per chunk, one array per field named "chunkNNNNN.<field>": review_ids flattened with offsets local to the chunk,
reviews_embedding with a has_reviews mask (zeros where the game has no reviews). A JSON "meta" entry (collection, model,
dimension, TEXT_VERSION, rows, chunks, ...) is written last. Format 1 (everything in one unprefixed chunk) is still readable.
"""

import json
import logging
import os
import time
import zipfile
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import numpy as np

from ml.config import Settings
from ml.text import TEXT_VERSION
from ml.vectors import DESC_VECTOR_FIELD, REVIEWS_VECTOR_FIELD, VECTOR_FIELD, GameVectorStore

log = logging.getLogger(__name__)

FORMAT_VERSION = 2
READABLE_FORMATS = {1, 2}
BATCH_SIZE = 500
CHUNK_SIZE = 20_000
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


def _chunk_prefix(index: int) -> str:
    return f"chunk{index:05d}."


def _to_arrays(rows: list[dict], dim: int) -> dict[str, np.ndarray]:
    """One Milvus batch → numpy right away (float32 instead of Python floats)."""
    count = len(rows)
    arrays: dict[str, np.ndarray] = {
        name: np.asarray([row.get(name) for row in rows], dtype=dtype) for name, dtype in SCALAR_FIELDS.items()
    }
    for name in (VECTOR_FIELD, DESC_VECTOR_FIELD):
        arrays[name] = np.asarray([row[name] for row in rows], dtype=np.float32).reshape(count, dim)
    reviews = np.zeros((count, dim), dtype=np.float32)
    has_reviews = np.zeros(count, dtype=bool)
    for i, row in enumerate(rows):
        vector = row.get(REVIEWS_VECTOR_FIELD)
        if vector is not None and len(vector) > 0:
            reviews[i] = vector
            has_reviews[i] = True
    arrays[REVIEWS_VECTOR_FIELD] = reviews
    arrays["has_reviews"] = has_reviews
    arrays["desc_hash"] = np.asarray([row.get("desc_hash") for row in rows], dtype="<U32")
    review_ids = [list(row.get("review_ids") or []) for row in rows]
    arrays["review_ids"] = np.asarray([i for ids in review_ids for i in ids], dtype=np.int64)
    arrays["review_ids_lengths"] = np.asarray([len(ids) for ids in review_ids], dtype=np.int64)
    return arrays


def _merge(batches: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    merged = {name: np.concatenate([b[name] for b in batches]) for name in batches[0]}
    lengths = merged.pop("review_ids_lengths")
    merged["review_ids_offsets"] = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return merged


def _write_array(archive: zipfile.ZipFile, name: str, array: np.ndarray) -> None:
    # The same member layout as np.savez_compressed, so np.load reads it lazily, member by member
    with archive.open(name + ".npy", "w", force_zip64=True) as f:
        np.lib.format.write_array(f, np.asanyarray(array), allow_pickle=False)


def export_vectors(settings: Settings, out: Path | None, collection: str | None = None,
                   chunk_size: int = CHUNK_SIZE) -> Path:
    store = _store(settings, collection)
    try:
        if not store.exists():
            raise BackupError(f"Collection {store.collection} doesn't exist")
        out = out or Path(f"milvus-{store.collection}-{datetime.now():%Y%m%dT%H%M}.npz")
        if out.exists():
            raise BackupError(f"{out} already exists")

        dim = settings.model.dimension
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp.npz")
        count = chunks = pending_rows = 0
        pending: list[dict[str, np.ndarray]] = []

        def flush() -> None:
            nonlocal chunks, pending_rows
            for name, array in _merge(pending).items():
                _write_array(archive, _chunk_prefix(chunks) + name, array)
            log.info("Wrote chunk %d (%d rows, %d total)", chunks, pending_rows, count)
            chunks += 1
            pending.clear()
            pending_rows = 0

        try:
            with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
                for rows in store.raw_rows(ALL_FIELDS, BATCH_SIZE):
                    if not rows:
                        continue
                    pending.append(_to_arrays(rows, dim))
                    pending_rows += len(rows)
                    count += len(rows)
                    if pending_rows >= chunk_size:
                        flush()
                if pending:
                    flush()
                meta = {
                    "format_version": FORMAT_VERSION,
                    "collection": store.collection,
                    "model": settings.model.name,
                    "dimension": dim,
                    "text_version": TEXT_VERSION,
                    "rows": count,
                    "chunks": chunks,
                    "exported_at": int(time.time()),
                }
                _write_array(archive, "meta", np.array(json.dumps(meta)))
            os.replace(tmp, out)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        log.info("Exported %d vectors in %d chunks from %s → %s (%.1f MB)",
                 count, chunks, store.collection, out, out.stat().st_size / 2**20)
        return out
    finally:
        store.close()


def _rows(data: np.lib.npyio.NpzFile, prefix: str) -> Iterator[list[dict]]:
    """One chunk of the archive (loaded only now) → batches of Milvus rows."""
    arrays = {name: data[prefix + name] for name in
              [*SCALAR_FIELDS, VECTOR_FIELD, DESC_VECTOR_FIELD, REVIEWS_VECTOR_FIELD, "has_reviews", "desc_hash",
               "review_ids", "review_ids_offsets"]}
    offsets = arrays["review_ids_offsets"]
    count = len(arrays["id"])
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
        yield rows


def import_vectors(settings: Settings, source: Path, collection: str | None = None, overwrite: bool = False) -> int:
    if not source.exists():
        raise BackupError(f"{source} not found")
    with np.load(source, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        if meta.get("format_version") not in READABLE_FORMATS:
            raise BackupError(f"Unsupported backup format {meta.get('format_version')}")
        if meta["dimension"] != settings.model.dimension or meta["model"] != settings.model.name:
            raise BackupError(
                f"Backup is for {meta['model']} (dim {meta['dimension']}), the config uses "
                f"{settings.model.name} (dim {settings.model.dimension})"
            )
        if meta["text_version"] != TEXT_VERSION:
            log.warning("Backup was made with TEXT_VERSION %s, current is %s: `ml stale` will list these vectors "
                        "as outdated", meta["text_version"], TEXT_VERSION)
        prefixes = [""] if meta["format_version"] == 1 else [_chunk_prefix(i) for i in range(meta["chunks"])]

        store = _store(settings, collection)
        try:
            store.ensure_collection()
            existing = store.count()
            if existing and not overwrite:
                raise BackupError(f"Collection {store.collection} already has {existing} vectors. "
                                  "Rows with the same id are replaced, others are kept. Pass --yes to continue")

            count = int(meta["rows"])
            imported = 0
            for chunk, prefix in enumerate(prefixes):
                for rows in _rows(data, prefix):
                    store.upsert_rows(rows)
                    imported += len(rows)
                log.info("Imported chunk %d / %d (%d / %d rows)", chunk + 1, len(prefixes), imported, count)
            if imported != count:
                raise BackupError(f"Backup meta says {count} rows, the chunks hold {imported}")
            log.info("Imported %d vectors from %s into %s (collection now has %d)",
                     count, source, store.collection, store.count())
            return count
        finally:
            store.close()
