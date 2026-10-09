"""Read-only access to the MariaDB game catalogue (the backend owns the schema)."""

from dataclasses import dataclass

import pymysql
import pymysql.cursors

from ml.config import DatabaseSettings


@dataclass(frozen=True)
class GameText:
    id: int
    title: str
    description: str


class GameRepository:
    """Not thread-safe: use one instance per thread."""

    def __init__(self, settings: DatabaseSettings) -> None:
        self._settings = settings
        self._connection: pymysql.Connection | None = None

    def _connect(self) -> pymysql.Connection:
        if self._connection is None:
            self._connection = pymysql.connect(
                host=self._settings.host,
                port=self._settings.port,
                user=self._settings.user,
                password=self._settings.password,
                database=self._settings.database,
                charset="utf8mb4",
                autocommit=True,  # every SELECT sees the latest committed data
                cursorclass=pymysql.cursors.DictCursor,
            )
        else:
            self._connection.ping(reconnect=True)
        return self._connection

    def get_games(self, ids: list[int]) -> list[GameText]:
        if not ids:
            return []
        placeholders = ", ".join(["%s"] * len(ids))
        with self._connect().cursor() as cursor:
            cursor.execute(
                f"SELECT id, title, description FROM games WHERE id IN ({placeholders}) ORDER BY id",
                ids,
            )
            return [GameText(row["id"], row["title"], row["description"]) for row in cursor.fetchall()]

    def get_game_ids(self, limit: int | None = None, after_id: int = 0) -> list[int]:
        """Game ids in ascending order, optionally only the first `limit` after `after_id`."""
        sql = "SELECT id FROM games WHERE id > %s ORDER BY id"
        params: list[int] = [after_id]
        if limit is not None:
            sql += " LIMIT %s"
            params.append(limit)
        with self._connect().cursor() as cursor:
            cursor.execute(sql, params)
            return [row["id"] for row in cursor.fetchall()]

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
