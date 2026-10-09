"""Game vectors in Milvus. Primary key = MariaDB games.id, so an upsert replaces a game's vector."""

import logging
import time

from pymilvus import DataType, MilvusClient

from ml.config import MilvusSettings, ModelSettings

log = logging.getLogger(__name__)

VECTOR_FIELD = "embedding"


class GameVectorStore:
    def __init__(self, settings: MilvusSettings, model: ModelSettings) -> None:
        self._client = MilvusClient(uri=settings.uri, token=settings.token)
        self._collection = settings.collection
        self._model = model

    @property
    def collection(self) -> str:
        return self._collection

    def ensure_collection(self) -> None:
        """Creates the collection if missing, checks the dimension and loads it."""
        if not self._client.has_collection(self._collection):
            self._create_collection()
            return

        info = self._client.describe_collection(self._collection)
        field = next(f for f in info["fields"] if f["name"] == VECTOR_FIELD)
        dim = int(field["params"]["dim"])
        if dim != self._model.dimension:
            raise ValueError(
                f"Collection {self._collection} has dim={dim}, model.dimension={self._model.dimension}. "
                "Use a new collection name for a different model/dimension"
            )
        self._client.load_collection(self._collection)

    def _create_collection(self) -> None:
        schema = self._client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
            description=f"Game embeddings, model {self._model.name}, dim {self._model.dimension}",
        )
        schema.add_field("id", DataType.INT64, is_primary=True, description="MariaDB games.id")
        schema.add_field(VECTOR_FIELD, DataType.FLOAT_VECTOR, dim=self._model.dimension)
        schema.add_field("text_version", DataType.INT32, description="ml.text.TEXT_VERSION used to build the text")
        schema.add_field("updated_at", DataType.INT64, description="Unix time of vectorization")

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name=VECTOR_FIELD,
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        # create_collection with index_params also loads the collection
        self._client.create_collection(self._collection, schema=schema, index_params=index_params)
        log.info("Created Milvus collection %s", self._collection)

    def upsert(self, vectors: dict[int, list[float]], text_version: int) -> None:
        if not vectors:
            return
        now = int(time.time())
        rows = [
            {"id": game_id, VECTOR_FIELD: vector, "text_version": text_version, "updated_at": now}
            for game_id, vector in vectors.items()
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
