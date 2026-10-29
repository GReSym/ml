"""Content-based recommendations from the user's rated games, searched in Milvus.

The default (POST /recommendations) was chosen by the offline evaluation (docs/evaluation.md): item-kNN on mean-centered
vectors plus a small popularity term. The averaged taste profile (`build_profile`) is kept as the evaluation's baseline.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, get_args

import numpy as np

from ml.vectors import GameVectorStore

if TYPE_CHECKING:
    from ml.catalogue import CatalogueStats

RATING_MIN, RATING_MAX = 1, 10
# How strongly disliked games push the profile away (0 = ignore dislikes)
DISLIKE_WEIGHT = 0.5
# Item-kNN: liked games searched (highest weight first), neighbours per search, best similarities averaged per candidate
KNN_QUERIES = 30
KNN_NEIGHBOURS = 50
KNN_TOP = 3
# User-facing knob of the popularity term: mainstream 0–100 → β = MAX_BETA · mainstream / 100 (score = kNN + β·popularity).
# 0 = pure similarity, the most niche; the default 10 (β 0.05); about 80 (β 0.4) is as accurate as "popular" in the evaluation
DEFAULT_MAINSTREAM = 10
MAX_BETA = 0.5
# Most popular games (the user hasn't rated) added to the kNN candidates when β > 0
POPULAR_CANDIDATES = 1000

# How disliked games change the kNN ranking (docs/evaluation.md, "Disliked games"):
# - centroid: score −= λ·cos(game, weighted mean of the dislikes)
# - knn: score −= λ·(best weighted similarity to a single disliked game)
# - margin: score −= λ·max(0, best dislike similarity − best like similarity): only games closer to a dislike than to any like
# - exclude: the nearest neighbours of the disliked games are never recommended (λ unused)
# - rocchio: every liked query moves away from the dislikes near it, l ← normalize(l − λ·Σ max(0, cos(l, d))·|w|·d),
#   the closed form of DEO-style query optimization (pull to positives, push from negatives) applied per liked game
DislikeMode = Literal["centroid", "knn", "margin", "exclude", "rocchio"]
DISLIKE_MODES: tuple[DislikeMode, ...] = get_args(DislikeMode)
PENALTY_MODES: tuple[DislikeMode, ...] = ("centroid", "knn", "margin")

# The user's dislike setting (backend `user_settings`): only the modes a user can understand, mapped to the ones above.
# None of them beat ignoring dislikes in the evaluation, so `ignore` is the default and the rest are opt-in.
UserDislikeMode = Literal["ignore", "penalize_closer", "penalize_similar", "hide_similar"]
USER_DISLIKE_MODES: dict[UserDislikeMode, DislikeMode | None] = {
    "ignore": None,
    "penalize_closer": "margin",
    "penalize_similar": "knn",
    "hide_similar": "exclude",
}
DEFAULT_USER_DISLIKE_MODE: UserDislikeMode = "ignore"
# Dislike strength 0–100: λ = MAX_DISLIKE_LAMBDA · strength / 100 for the penalty modes,
# neighbours hidden per dislike = strength for hide_similar (the default 50 = KNN_NEIGHBOURS, the evaluated behaviour)
DEFAULT_DISLIKE_STRENGTH = 50
MAX_DISLIKE_LAMBDA = 1.0


@dataclass(frozen=True)
class DislikeParams:
    mode: DislikeMode | None = None  # None = dislikes are ignored
    weight: float = 0.0  # λ of the penalty modes, γ of rocchio
    neighbours: int = KNN_NEIGHBOURS  # exclude: nearest games hidden per dislike

    @property
    def active(self) -> bool:
        if self.mode == "exclude":
            return self.neighbours > 0
        return self.mode is not None and self.weight > 0


def dislike_params(mode: UserDislikeMode, strength: int) -> DislikeParams:
    """The user's dislike setting → the algorithm's parameters. Strength 0 means no effect in every mode."""
    internal = USER_DISLIKE_MODES[mode]
    if internal is None:
        return DislikeParams()
    if internal == "exclude":
        return DislikeParams(internal, neighbours=strength)
    return DislikeParams(internal, weight=MAX_DISLIKE_LAMBDA * strength / 100)


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


def _weighted_matrix(ratings: list[Rating], vectors: dict[int, list[float]], sign: int) -> tuple[np.ndarray, np.ndarray]:
    """(vectors n×d, |weights| n) of the rated games with a vector and a weight of the given sign, highest |weight| first."""
    pairs = [(abs(w), vectors[r.game_id]) for r in ratings
             if r.game_id in vectors and (w := rating_weight(r)) * sign > 0]
    pairs.sort(key=lambda pair: -pair[0])
    if not pairs:
        return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=np.float32)
    return (np.asarray([vector for _, vector in pairs], dtype=np.float32),
            np.asarray([weight for weight, _ in pairs], dtype=np.float32))


def liked_matrix(ratings: list[Rating], vectors: dict[int, list[float]]) -> tuple[np.ndarray, np.ndarray]:
    """(vectors n×d, weights n) of the liked games that have a vector, highest weight first."""
    return _weighted_matrix(ratings, vectors, 1)


def disliked_matrix(ratings: list[Rating], vectors: dict[int, list[float]]) -> tuple[np.ndarray, np.ndarray]:
    """(vectors n×d, |weights| n) of the disliked games that have a vector, strongest dislike first."""
    return _weighted_matrix(ratings, vectors, -1)


def knn_scores(candidates: np.ndarray, liked: np.ndarray, weights: np.ndarray, top: int = KNN_TOP) -> np.ndarray:
    """Item-kNN score of every candidate row: the mean of its `top` best weighted similarities to single liked games.

    Unlike the averaged profile, a game close to one part of the user's taste scores high even when the user
    likes several unrelated genres (the average of those lands between them, near no game the user would pick).
    Disliked games are handled separately (`dislike_penalty`, `rocchio_shift`, `dislike_neighbours`).
    """
    sims = (candidates @ liked.T) * weights
    top = min(top, sims.shape[1])
    return np.partition(sims, sims.shape[1] - top, axis=1)[:, -top:].mean(axis=1)


def dislike_penalty(
    candidates: np.ndarray,
    disliked: np.ndarray,
    dweights: np.ndarray,
    mode: DislikeMode,
    liked: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Penalty of every candidate row for modes centroid/knn/margin (see DislikeMode); zeros without dislikes.

    `margin` needs the liked rows: a game as close to a liked game as to a disliked one isn't penalized, because on Steam
    a dislike is usually one disappointing game from a genre the user otherwise plays.
    """
    if not len(disliked):
        return np.zeros(len(candidates), dtype=np.float32)
    if mode == "centroid":
        centroid = _normalize(dweights @ disliked)
        return candidates @ centroid if centroid is not None else np.zeros(len(candidates), dtype=np.float32)
    nearest_dislike = ((candidates @ disliked.T) * dweights).max(axis=1)
    if mode == "knn":
        return nearest_dislike
    if mode == "margin":
        nearest_like = ((candidates @ liked.T) * weights).max(axis=1)
        return np.maximum(0, nearest_dislike - nearest_like)
    raise ValueError(f"{mode} isn't a penalty mode")


def rocchio_shift(liked: np.ndarray, disliked: np.ndarray, dweights: np.ndarray, gamma: float) -> np.ndarray:
    """Liked rows moved away from the disliked games near each of them, normalized again (unchanged without dislikes)."""
    if not len(disliked) or not len(liked) or gamma == 0:
        return liked
    pull = np.maximum(0, liked @ disliked.T) * dweights
    shifted = liked - gamma * (pull @ disliked)
    norms = np.linalg.norm(shifted, axis=1, keepdims=True)
    return shifted / np.where(norms > 0, norms, 1)


def dislike_neighbours(
    store: GameVectorStore, disliked: np.ndarray, exclude_ids: list[int], neighbours: int = KNN_NEIGHBOURS
) -> set[int]:
    """The `neighbours` nearest games of each of the strongest dislikes (mode `exclude`), one Milvus request."""
    if not len(disliked) or neighbours <= 0:
        return set()
    hits = store.search_many(disliked[:KNN_QUERIES].tolist(), neighbours, exclude_ids)
    return {game_id for query in hits for game_id, _ in query}


def center(vectors: np.ndarray, mean: np.ndarray | None) -> np.ndarray:
    """Rows of `vectors` minus the catalogue mean, normalized again (`mean` None = unchanged).

    Embeddings share a common direction, so the plain cosine favours games near the catalogue's centre ("hubs" that
    are close to every profile). Centering removes it and leaves what distinguishes the games.
    """
    if mean is None:
        return vectors
    shifted = vectors - mean
    norms = np.linalg.norm(shifted, axis=1, keepdims=True)
    return shifted / np.where(norms > 0, norms, 1)


def knn_candidates(store: GameVectorStore, liked: np.ndarray, exclude_ids: list[int]) -> list[int]:
    """The nearest neighbours of the highest-weighted liked games (`liked` rows, best first), one Milvus request.

    With centered `liked` rows the stored (uncentered) vectors still rank right: cosine(q, v) differs from q·(v − mean)
    by a constant per query. Only the per-game norm of v − mean is missing, so the caller re-scores the candidates.
    """
    hits = store.search_many(liked[:KNN_QUERIES].tolist(), KNN_NEIGHBOURS, exclude_ids)
    return sorted({game_id for query in hits for game_id, _ in query})


def mainstream_beta(mainstream: int) -> float:
    return MAX_BETA * mainstream / 100


def recommend(
    store: GameVectorStore,
    stats: "CatalogueStats",
    ratings: list[Rating],
    limit: int,
    mainstream: int = DEFAULT_MAINSTREAM,
    exclude_ids: list[int] | None = None,
    dislikes: DislikeParams = DislikeParams(),
) -> Recommendations:
    """Centered item-kNN + β·popularity, with the user's dislike handling (`dislikes`, ignored by default).

    Rated games are never recommended; `exclude_ids` hides more.
    """
    rated_ids = [rating.game_id for rating in ratings]
    rated_vectors = store.get_vectors(rated_ids)
    liked, weights = liked_matrix(ratings, rated_vectors)
    if not len(liked):
        return Recommendations(items=[], used_ratings=0)
    liked = center(liked, stats.mean)
    excluded = set(rated_ids) | set(exclude_ids or [])

    disliked, dweights = disliked_matrix(ratings, rated_vectors) if dislikes.active else (np.empty((0, 0)), None)
    if len(disliked):
        disliked = center(disliked, stats.mean)
        if dislikes.mode == "rocchio":
            liked = rocchio_shift(liked, disliked, dweights, dislikes.weight)
        elif dislikes.mode == "exclude":
            excluded |= dislike_neighbours(store, disliked, sorted(excluded), dislikes.neighbours)

    fetched = store.get_vectors(knn_candidates(store, liked, sorted(excluded)))
    pool = list(fetched)
    rows = [center(np.asarray([fetched[i] for i in pool], dtype=np.float32).reshape(len(pool), -1), stats.mean)]
    beta = mainstream_beta(mainstream)
    if beta > 0:
        taken = excluded | set(pool)
        extra = [(row, game_id) for row, game_id in enumerate(stats.popular_ids) if game_id not in taken]
        extra = extra[:POPULAR_CANDIDATES]
        pool += [game_id for _, game_id in extra]
        rows.append(stats.popular_vectors[[row for row, _ in extra]].reshape(len(extra), -1))
    used = len(liked) + len(disliked)
    if not pool:
        return Recommendations(items=[], used_ratings=used)

    matrix = np.concatenate(rows)
    scores = knn_scores(matrix, liked, weights) + beta * stats.popularity(pool)
    if len(disliked) and dislikes.mode in PENALTY_MODES:
        scores -= dislikes.weight * dislike_penalty(matrix, disliked, dweights, dislikes.mode, liked, weights)
    order = np.argsort(-scores, kind="stable")[:limit]
    return Recommendations(items=[(pool[i], float(scores[i])) for i in order], used_ratings=used)


def popularity(counts: np.ndarray, max_count: int) -> np.ndarray:
    """Steam recommendations count → [0, 1] on a log scale (the most recommended game = 1, none = 0)."""
    return np.log1p(counts) / np.log1p(max(max_count, 1))


def similar(store: GameVectorStore, game_id: int, limit: int) -> list[tuple[int, float]] | None:
    """Games closest to one game. None if the game has no vector."""
    vector = store.get_vectors([game_id]).get(game_id)
    if vector is None:
        return None
    return store.search(vector, limit, exclude_ids=[game_id])
