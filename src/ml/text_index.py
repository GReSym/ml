"""Keyword (BM25) index of game titles and descriptions in Milvus, used by ml.search.

A separate collection from the game vectors: it needs no model (Milvus tokenizes the text and computes the BM25
sparse vectors itself, on insert and on query), so it's filled from MariaDB in minutes and can be rebuilt any time
without touching the hours-long dense vectorization.
"""

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass

from pymilvus import DataType, Function, FunctionType, MilvusClient

from ml.config import MilvusSettings
from ml.db import GameRepository, GameText
from ml.text import build_game_text, html_to_text, text_hash

log = logging.getLogger(__name__)

# Bump when the analyzer or the indexed text changes: rows with an older version are rebuilt by
# `ml index-text --skip-existing`. An analyzer change also needs a new collection (it's part of the schema).
TEXT_INDEX_VERSION = 1

TITLE_FIELD = "title"
TEXT_FIELD = "text"
TITLE_SPARSE_FIELD = "title_sparse"
TEXT_SPARSE_FIELD = "text_sparse"
MAX_TITLE_CHARS = 1024
MAX_TEXT_BYTES = 65_000  # VARCHAR(65535) counts UTF-8 bytes; the longest description is ~41k chars (~82 KB of Cyrillic)

# Content is Russian with Latin titles: Russian stop words are dropped and words are stemmed
# ("открытым миром" → "открыт", "мир"); Latin words pass through lowercased.
ANALYZER = {
    "tokenizer": "standard",
    "filter": ["lowercase", {"type": "stop", "stop_words": ["_russian_"]}, {"type": "stemmer", "language": "russian"}],
}


@dataclass(frozen=True)
class GameTextRow:
    id: int
    title: str
    text: str
    desc_hash: str  # same hash as the dense vector's snapshot (ml.text.text_hash of the embedded text)


def _truncate_bytes(text: str, limit: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    return encoded[:limit].decode("utf-8", errors="ignore")


def build_text_row(game: GameText) -> GameTextRow:
    return GameTextRow(
        id=game.id,
        title=game.title.strip()[:MAX_TITLE_CHARS],
        text=_truncate_bytes(html_to_text(game.description), MAX_TEXT_BYTES),
        desc_hash=text_hash(build_game_text(game)),
    )


class GameTextStore:
    def __init__(self, settings: MilvusSettings) -> None:
        self._client = MilvusClient(uri=settings.uri, token=settings.token)
        self._collection = settings.text_collection

    @property
    def collection(self) -> str:
        return self._collection

    def ensure_collection(self) -> None:
        if not self._client.has_collection(self._collection):
            self._create_collection()
            return
        self._client.load_collection(self._collection)

    def _create_collection(self) -> None:
        schema = self._client.create_schema(
            auto_id=False, enable_dynamic_field=False, description="BM25 index of game titles and descriptions",
        )
        schema.add_field("id", DataType.INT64, is_primary=True, description="MariaDB games.id")
        schema.add_field(TITLE_FIELD, DataType.VARCHAR, max_length=MAX_TITLE_CHARS,
                         enable_analyzer=True, analyzer_params=ANALYZER)
        schema.add_field(TEXT_FIELD, DataType.VARCHAR, max_length=65535, enable_analyzer=True, analyzer_params=ANALYZER,
                         description="description without HTML")
        schema.add_field(TITLE_SPARSE_FIELD, DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(TEXT_SPARSE_FIELD, DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("desc_hash", DataType.VARCHAR, max_length=32, description="hash of title + description text")
        schema.add_field("text_version", DataType.INT32, description="TEXT_INDEX_VERSION used")
        schema.add_field("updated_at", DataType.INT64, description="Unix time of indexing")
        for source, target in ((TITLE_FIELD, TITLE_SPARSE_FIELD), (TEXT_FIELD, TEXT_SPARSE_FIELD)):
            schema.add_function(Function(
                name=f"{source}_bm25", input_field_names=[source], output_field_names=[target],
                function_type=FunctionType.BM25,
            ))

        index_params = self._client.prepare_index_params()
        for field in (TITLE_SPARSE_FIELD, TEXT_SPARSE_FIELD):
            index_params.add_index(field_name=field, index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
        self._client.create_collection(self._collection, schema=schema, index_params=index_params)
        log.info("Created Milvus collection %s", self._collection)

    def upsert(self, rows: list[GameTextRow]) -> None:
        if not rows:
            return
        now = int(time.time())
        self._client.upsert(self._collection, data=[
            {
                "id": r.id,
                TITLE_FIELD: r.title,
                TEXT_FIELD: r.text,
                "desc_hash": r.desc_hash,
                "text_version": TEXT_INDEX_VERSION,
                "updated_at": now,
            }
            for r in rows
        ])

    def delete(self, ids: list[int]) -> None:
        if ids:
            self._client.delete(self._collection, ids=ids)

    def existing_hashes(self, ids: list[int]) -> dict[int, str]:
        """desc_hash of the given games that are indexed with the current TEXT_INDEX_VERSION."""
        if not ids:
            return {}
        rows = self._client.query(
            self._collection,
            filter=f"id in {list(ids)} and text_version == {TEXT_INDEX_VERSION}",
            output_fields=["id", "desc_hash"],
        )
        return {row["id"]: row["desc_hash"] for row in rows}

    def search(self, query: str, limit: int) -> dict[str, list[int]]:
        """Ranked game ids per field (title, text), best first. A query of only stop words matches nothing."""
        result = self._client.search(
            self._collection,
            data=[query],
            anns_field=TITLE_SPARSE_FIELD,
            limit=limit,
            search_params={"metric_type": "BM25"},
        )
        titles = [hit["id"] for hit in result[0]]
        result = self._client.search(
            self._collection,
            data=[query],
            anns_field=TEXT_SPARSE_FIELD,
            limit=limit,
            search_params={"metric_type": "BM25"},
        )
        return {TITLE_FIELD: titles, TEXT_FIELD: [hit["id"] for hit in result[0]]}

    def count(self) -> int:
        rows = self._client.query(
            self._collection, filter="", output_fields=["count(*)"], consistency_level="Strong"
        )
        return int(rows[0]["count(*)"])

    def close(self) -> None:
        self._client.close()


def _chunks(ids: list[int], size: int) -> Iterator[list[int]]:
    for start in range(0, len(ids), size):
        yield ids[start:start + size]


def run_index(settings: MilvusSettings, games: GameRepository, ids: list[int] | None, limit: int | None,
              skip_existing: bool, chunk: int = 500) -> None:
    """Indexes the given ids, or the first `limit` games by id (all games if limit is None). No model needed.

    With skip_existing, games whose indexed text is current (same hash and TEXT_INDEX_VERSION) are skipped,
    so the command doubles as "index whatever is missing or changed". Games missing in MariaDB are removed.
    """
    store = GameTextStore(settings)
    try:
        store.ensure_collection()
        if ids is None:
            ids = games.get_game_ids(limit=limit)
        log.info("Indexing %d games into %s", len(ids), store.collection)

        started = time.monotonic()
        done = skipped = deleted = 0
        for batch in _chunks(ids, chunk):
            found = games.get_games(batch)
            rows = [build_text_row(game) for game in found]
            if skip_existing:
                existing = store.existing_hashes([r.id for r in rows])
                fresh = [r for r in rows if existing.get(r.id) != r.desc_hash]
                skipped += len(rows) - len(fresh)
                rows = fresh
            store.upsert(rows)
            found_ids = {game.id for game in found}
            missing = [game_id for game_id in batch if game_id not in found_ids]
            store.delete(missing)
            done += len(rows)
            deleted += len(missing)
            log.info("%d/%d games: %d indexed, %d skipped, %d deleted (%.0f s)",
                     min(done + skipped + deleted, len(ids)), len(ids), done, skipped, deleted,
                     time.monotonic() - started)
    finally:
        store.close()
