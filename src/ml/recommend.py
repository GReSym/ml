"""Content-based recommendations: a taste profile built from the user's rated games, searched in Milvus."""

from dataclasses import dataclass

import numpy as np

from ml.vectors import GameVectorStore

RATING_MIN, RATING_MAX = 1, 10
# How strongly disliked games push the profile away (0 = ignore dislikes)
DISLIKE_WEIGHT = 0.5


@dataclass(frozen=True)
class Rating:
    game_id: int
    rating: int | None = None
    recommend: bool | None = None


@dataclass
class Recommendations:
    items: list[tuple[int, float]]
    used_ratings: int  # rated games that had a vector and a non-neutral weight


def rating_weight(rating: Rating) -> float:
    """Maps a rating to [-1, 1]: 10 → 1, 1 → -1, 5.5 → 0. Falls back to recommend (±1)."""
    if rating.rating is not None:
        middle = (RATING_MIN + RATING_MAX) / 2
        return (rating.rating - middle) / (RATING_MAX - middle)
    if rating.recommend is not None:
        return 1.0 if rating.recommend else -1.0
    return 0.0


def _normalize(vector: np.ndarray) -> np.ndarray | None:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else None


def build_profile(ratings: list[Rating], vectors: dict[int, list[float]]) -> tuple[np.ndarray | None, int]:
    """Weighted mean of liked games' vectors minus a share of the disliked ones.

    Returns (profile, number of ratings used). No liked games with vectors → no profile.
    """
    liked: list[np.ndarray] = []
    disliked: list[np.ndarray] = []
    for rating in ratings:
        vector = vectors.get(rating.game_id)
        weight = rating_weight(rating)
        if vector is None or weight == 0:
            continue
        (liked if weight > 0 else disliked).append(abs(weight) * np.asarray(vector, dtype=np.float32))

    if not liked:
        return None, len(disliked)

    profile = _normalize(np.sum(liked, axis=0))
    if profile is not None and disliked:
        dislike = _normalize(np.sum(disliked, axis=0))
        if dislike is not None:
            profile = _normalize(profile - DISLIKE_WEIGHT * dislike)
    return profile, len(liked) + len(disliked)


def recommend(store: GameVectorStore, ratings: list[Rating], limit: int) -> Recommendations:
    rated_ids = [rating.game_id for rating in ratings]
    profile, used = build_profile(ratings, store.get_vectors(rated_ids))
    if profile is None:
        return Recommendations(items=[], used_ratings=used)
    return Recommendations(items=store.search(profile.tolist(), limit, exclude_ids=rated_ids), used_ratings=used)


def similar(store: GameVectorStore, game_id: int, limit: int) -> list[tuple[int, float]] | None:
    """Games closest to one game. None if the game has no vector."""
    vector = store.get_vectors([game_id]).get(game_id)
    if vector is None:
        return None
    return store.search(vector, limit, exclude_ids=[game_id])
