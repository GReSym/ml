"""Game vectors in Milvus. Primary key = MariaDB games.id, so an upsert replaces a game's vector.

Each row keeps the combined vector used for search (`embedding`), its two components (description and
reviews, so the review weight can be retuned without re-embedding) and a snapshot of what it was built from
(used by ml.staleness to find games whose vector is out of date).
"""

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass

from pymilvus import DataType, MilvusClient

from ml.config import MilvusSettings, ModelSettings

log = logging.getLogger(__name__)

VECTOR_FIELD = "embedding"
DESC_VECTOR_FIELD = "desc_embedding"
REVIEWS_VECTOR_FIELD = "reviews_embedding"
MAX_REVIEW_IDS = 64  # ARRAY capacity, also the upper limit of reviews.max_selected

# Snapshot fields, read by the staleness scan (no vectors)
SNAPSHOT_FIELDS = ["id", "text_version", "updated_at", "desc_hash", "review_ids", "steam_reviews_total",
                   "steam_positive_share"]


@dataclass(frozen=True)
class GameVector:
    id: int
    embedding: list[float]
    desc_embedding: list[float]
    reviews_embedding: list[float] | None  # None: the game has no usable reviews
    reviews_weight: float
    text_version: int
    desc_hash: str
    review_ids: list[int]
    steam_reviews_total: int | None
    steam_positive_share: float | None


@dataclass(frozen=True)
class VectorSnapshot:
    id: int
    text_version: int
    updated_at: int
    desc_hash: str
    review_ids: list[int]
    steam_reviews_total: int | None
    steam_positive_share: float | None


class GameVectorStore:
    def __init__(self, settings: MilvusSettings, model: ModelSettings) -> None:
        self._client = MilvusClient(uri=settings.uri, token=settings.token)
        self._collection = settings.collection
        self._model = model

    @property
    def collection(self) -> str:
        return self._collection

    def ensure_collection(self) -> None:
        """Creates the collection if missing, checks the dimension and the schema, and loads it."""
        if not self._client.has_collection(self._collection):
            self._create_collection()
            return

        info = self._client.describe_collection(self._collection)
        fields = {f["name"]: f for f in info["fields"]}
        missing = [name for name in (DESC_VECTOR_FIELD, REVIEWS_VECTOR_FIELD, *SNAPSHOT_FIELDS) if name not in fields]
        if missing:
            raise ValueError(
                f"Collection {self._collection} has an old schema (no {', '.join(missing)}). "
                "Set milvus.collection to a new name"
            )
        dim = int(fields[VECTOR_FIELD]["params"]["dim"])
        if dim != self._model.dimension:
            raise ValueError(
                f"Collection {self._collection} has dim={dim}, model.dimension={self._model.dimension}. "
                "Use a new collection name for a different model/dimension"
            )
        self._client.load_collection(self._collection)

    def _create_collection(self) -> None:
        dim = self._model.dimension
        schema = self._client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
            description=f"Game embeddings, model {self._model.name}, dim {dim}",
        )
        schema.add_field("id", DataType.INT64, is_primary=True, description="MariaDB games.id")
        schema.add_field(VECTOR_FIELD, DataType.FLOAT_VECTOR, dim=dim,
                         description="normalize(desc + reviews_weight * reviews), used for search")
        # Components are only read back by id, never searched: memory-mapped, so they don't take RAM
        schema.add_field(DESC_VECTOR_FIELD, DataType.FLOAT_VECTOR, dim=dim, mmap_enabled=True,
                         description="title + description")
        schema.add_field(REVIEWS_VECTOR_FIELD, DataType.FLOAT_VECTOR, dim=dim, nullable=True, mmap_enabled=True,
                         description="normalized mean of the selected reviews; null without reviews")
        schema.add_field("reviews_weight", DataType.FLOAT, description="weight of reviews_embedding")
        schema.add_field("text_version", DataType.INT32, description="ml.text.TEXT_VERSION used to build the text")
        schema.add_field("updated_at", DataType.INT64, description="Unix time of vectorization")
        schema.add_field("desc_hash", DataType.VARCHAR, max_length=32, description="hash of the description text")
        schema.add_field("review_ids", DataType.ARRAY, element_type=DataType.INT64, max_capacity=MAX_REVIEW_IDS,
                         description="MariaDB reviews.id that were embedded")
        schema.add_field("steam_reviews_total", DataType.INT64, description="Steam review count, -1 unknown")
        schema.add_field("steam_positive_share", DataType.FLOAT, description="Steam positive share, -1 unknown")

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name=VECTOR_FIELD,
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        # Every vector field needs an index to load the collection. FLAT builds nothing extra
        for field in (DESC_VECTOR_FIELD, REVIEWS_VECTOR_FIELD):
            index_params.add_index(field_name=field, index_type="FLAT", metric_type="COSINE")
        # create_collection with index_params also loads the collection
        self._client.create_collection(self._collection, schema=schema, index_params=index_params)
        log.info("Created Milvus collection %s", self._collection)

    def upsert(self, vectors: list[GameVector]) -> None:
        if not vectors:
            return
        now = int(time.time())
        rows = [
            {
                "id": v.id,
                VECTOR_FIELD: v.embedding,
                DESC_VECTOR_FIELD: v.desc_embedding,
                REVIEWS_VECTOR_FIELD: v.reviews_embedding,
                "reviews_weight": v.reviews_weight,
                "text_version": v.text_version,
                "updated_at": now,
                "desc_hash": v.desc_hash,
                "review_ids": v.review_ids,
                "steam_reviews_total": -1 if v.steam_reviews_total is None else v.steam_reviews_total,
                "steam_positive_share": -1.0 if v.steam_positive_share is None else v.steam_positive_share,
            }
            for v in vectors
        ]
        self._client.upsert(self._collection, data=rows)

    def delete(self, ids: list[int]) -> None:
        if ids:
            self._client.delete(self._collection, ids=ids)

    def existing_ids(self, ids: list[int], text_version: int) -> set[int]:
        """Which of `ids` already have a vector built with `text_version`."""
        if not ids:
            return set()
        rows = self._client.query(
            self._collection,
            filter=f"id in {list(ids)} and text_version == {text_version}",
            output_fields=["id"],
        )
        return {row["id"] for row in rows}

    def snapshots(self, batch_size: int = 2000) -> Iterator[VectorSnapshot]:
        """What every stored vector was built from (no vectors are read)."""
        iterator = self._client.query_iterator(
            self._collection, batch_size=batch_size, filter="", output_fields=SNAPSHOT_FIELDS,
            consistency_level="Strong",
        )
        try:
            while rows := iterator.next():
                for row in rows:
                    total = int(row["steam_reviews_total"])
                    share = float(row["steam_positive_share"])
                    yield VectorSnapshot(
                        id=int(row["id"]),
                        text_version=int(row["text_version"]),
                        updated_at=int(row["updated_at"]),
                        desc_hash=row["desc_hash"],
                        review_ids=[int(i) for i in row["review_ids"]],
                        steam_reviews_total=None if total < 0 else total,
                        steam_positive_share=None if share < 0 else share,
                    )
        finally:
            iterator.close()

    def get_vectors(self, ids: list[int]) -> dict[int, list[float]]:
        if not ids:
            return {}
        rows = self._client.get(self._collection, ids=list(ids), output_fields=[VECTOR_FIELD])
        return {row["id"]: list(row[VECTOR_FIELD]) for row in rows}

    def search(self, vector: list[float], limit: int, exclude_ids: list[int]) -> list[tuple[int, float]]:
        """Nearest games by cosine similarity, as (game_id, score), best first."""
        result = self._client.search(
            self._collection,
            data=[vector],
            anns_field=VECTOR_FIELD,
            limit=limit,
            filter=f"id not in {list(exclude_ids)}" if exclude_ids else "",
            search_params={"metric_type": "COSINE", "params": {"ef": max(64, limit * 2)}},
        )
        return [(hit["id"], float(hit["distance"])) for hit in result[0]]

    def count(self) -> int:
        rows = self._client.query(
            self._collection, filter="", output_fields=["count(*)"], consistency_level="Strong"
        )
        return int(rows[0]["count(*)"])

    def close(self) -> None:
        self._client.close()
