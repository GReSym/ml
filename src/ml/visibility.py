"""Live catalogue exclusions for user-facing ML operations (no index rebuild required)."""

from ml.config import DatabaseSettings
from ml.db import GameRepository


def excluded_game_ids(settings: DatabaseSettings, hide_nsfw: bool) -> list[int]:
    if not hide_nsfw:
        return []
    games = GameRepository(settings)
    try:
        return games.get_nsfw_game_ids()
    finally:
        games.close()
