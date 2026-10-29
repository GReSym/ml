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
    steam_positive: int | None = None  # Steam review totals, all languages (steam_source)
    steam_negative: int | None = None


@dataclass(frozen=True)
class ReviewRow:
    id: int
    game_id: int
    content: str | None
    recommended: bool | None
    weighted_vote_score: float | None
    votes_up: int | None


@dataclass(frozen=True)
class GameChangeInfo:
    """Cheap per-game state used to find stale vectors. Times are unix seconds (UTC)."""
    id: int
    updated_at: float
    reviews_updated_at: float | None  # newest reviews.updated_at, None without reviews
    steam_positive: int | None
    steam_negative: int | None


@dataclass(frozen=True)
class SteamUserGame:
    game_id: int
    playtime_minutes: int
    recommended: bool | None  # the user's own Steam review, None without one


@dataclass(frozen=True)
class SteamUserData:
    """An anonymized Steam user collected for offline evaluation (steam_users + steam_user_games)."""
    id: int
    name: str
    games: list[SteamUserGame]


@dataclass(frozen=True)
class EvaluationRunRow:
    id: int
    status: str
    parameters: str  # JSON


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
                # The backend stores UTC datetimes, so UNIX_TIMESTAMP() must not shift them
                init_command="SET time_zone = '+00:00'",
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
                "SELECT g.id, g.title, g.description, s.reviews_positive, s.reviews_negative FROM games g "
                f"LEFT JOIN steam_source s ON s.game_id = g.id WHERE g.id IN ({placeholders}) ORDER BY g.id",
                ids,
            )
            return [
                GameText(row["id"], row["title"], row["description"], row["reviews_positive"], row["reviews_negative"])
                for row in cursor.fetchall()
            ]

    def get_reviews(self, game_ids: list[int]) -> dict[int, list[ReviewRow]]:
        """All reviews of the given games, grouped by game id."""
        result: dict[int, list[ReviewRow]] = {}
        if not game_ids:
            return result
        placeholders = ", ".join(["%s"] * len(game_ids))
        with self._connect().cursor() as cursor:
            cursor.execute(
                "SELECT id, game_id, content, recommended, weighted_vote_score, votes_up FROM reviews "
                f"WHERE game_id IN ({placeholders}) ORDER BY id",
                game_ids,
            )
            for row in cursor.fetchall():
                result.setdefault(row["game_id"], []).append(
                    ReviewRow(
                        row["id"],
                        row["game_id"],
                        row["content"],
                        None if row["recommended"] is None else bool(row["recommended"]),
                        row["weighted_vote_score"],
                        row["votes_up"],
                    )
                )
        return result

    def get_change_info(self) -> list[GameChangeInfo]:
        """Every game with its update times and Steam review totals (one pass, no texts)."""
        with self._connect().cursor() as cursor:
            cursor.execute(
                "SELECT g.id, UNIX_TIMESTAMP(g.updated_at) AS updated_at, "
                "UNIX_TIMESTAMP(r.reviews_updated_at) AS reviews_updated_at, "
                "s.reviews_positive, s.reviews_negative "
                "FROM games g "
                "LEFT JOIN steam_source s ON s.game_id = g.id "
                "LEFT JOIN (SELECT game_id, MAX(updated_at) AS reviews_updated_at FROM reviews "
                "           WHERE game_id IS NOT NULL GROUP BY game_id) r ON r.game_id = g.id "
                "ORDER BY g.id"
            )
            return [
                GameChangeInfo(
                    row["id"],
                    float(row["updated_at"] or 0),
                    None if row["reviews_updated_at"] is None else float(row["reviews_updated_at"]),
                    row["reviews_positive"],
                    row["reviews_negative"],
                )
                for row in cursor.fetchall()
            ]

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

    def get_steam_app_ids(self) -> dict[int, int]:
        """Steam app id → game id for every game from Steam."""
        with self._connect().cursor() as cursor:
            cursor.execute("SELECT steam_app_id, game_id FROM steam_source WHERE game_id IS NOT NULL")
            return {row["steam_app_id"]: row["game_id"] for row in cursor.fetchall()}

    def get_popular_game_ids(self, limit: int) -> list[int]:
        """Most recommended games first: the same order as the API's popular fallback."""
        with self._connect().cursor() as cursor:
            cursor.execute(
                "SELECT g.id FROM games g LEFT JOIN steam_source s ON s.game_id = g.id "
                "ORDER BY s.recommendations_count IS NULL, s.recommendations_count DESC, g.id LIMIT %s",
                [limit],
            )
            return [row["id"] for row in cursor.fetchall()]

    def get_recommendation_counts(self) -> dict[int, int]:
        """Game id → Steam recommendations count, for games that have one (the popularity used by the hybrid)."""
        with self._connect().cursor() as cursor:
            cursor.execute(
                "SELECT game_id, recommendations_count FROM steam_source "
                "WHERE game_id IS NOT NULL AND recommendations_count IS NOT NULL"
            )
            return {row["game_id"]: row["recommendations_count"] for row in cursor.fetchall()}

    def get_game_tags(self) -> tuple[list[int], list[int]]:
        """(game ids, tag ids) of every game_tags row, ~3.2 M pairs, streamed without per-row dicts."""
        game_ids: list[int] = []
        tag_ids: list[int] = []
        with self._connect().cursor(pymysql.cursors.SSCursor) as cursor:
            cursor.execute("SELECT game_id, tag_id FROM game_tags")
            while rows := cursor.fetchmany(100_000):
                for game_id, tag_id in rows:
                    game_ids.append(game_id)
                    tag_ids.append(tag_id)
        return game_ids, tag_ids

    def get_titles(self, ids: list[int]) -> dict[int, str]:
        if not ids:
            return {}
        placeholders = ", ".join(["%s"] * len(ids))
        with self._connect().cursor() as cursor:
            cursor.execute(f"SELECT id, title FROM games WHERE id IN ({placeholders})", ids)
            return {row["id"]: row["title"] for row in cursor.fetchall()}

    def get_steam_users(self) -> list[SteamUserData]:
        """Every collected Steam user with their games (in our catalogue), by user id."""
        with self._connect().cursor() as cursor:
            cursor.execute("SELECT id, name FROM steam_users ORDER BY id")
            users = {row["id"]: (row["name"], []) for row in cursor.fetchall()}
            cursor.execute(
                "SELECT steam_user_id, game_id, playtime_minutes, recommended FROM steam_user_games "
                "ORDER BY steam_user_id, game_id"
            )
            for row in cursor.fetchall():
                user = users.get(row["steam_user_id"])
                if user is not None:
                    recommended = None if row["recommended"] is None else bool(row["recommended"])
                    user[1].append(SteamUserGame(row["game_id"], row["playtime_minutes"], recommended))
        return [SteamUserData(user_id, name, games) for user_id, (name, games) in users.items()]

    def get_evaluation_run(self, run_id: int) -> EvaluationRunRow | None:
        with self._connect().cursor() as cursor:
            cursor.execute("SELECT id, status, parameters FROM evaluation_runs WHERE id = %s", [run_id])
            row = cursor.fetchone()
        return None if row is None else EvaluationRunRow(row["id"], row["status"], row["parameters"])

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
