"""Offline evaluation of recommendations on collected Steam users (leave-k-out, see docs/evaluation.md).

For every user some liked games are hidden (test), recommendations are built from the rest (train), and the
metrics measure how many hidden games come back in the top K. Strategies compared:
- gresym: the averaged taste profile searched in Milvus (POST /recommendations until 2026-10-27),
- knn: item-kNN over single liked games (recommend.knn_candidates + knn_scores),
- hybrid-β: gresym's similarity + β·popularity, re-ranking gresym's nearest games together with the most popular ones,
- knn-hybrid-β: the same with the item-kNN score (with `center`, what POST /recommendations does: recommend.recommend),
- optionally tag-only strategies (`tags`): tags-top (the user's 10 strongest tags, ties by popularity), tags (TF-IDF tag
  profile, cosine over the whole catalogue), tags-hybrid-β (+ β·popularity),
- popular (the API's fallback) and random games as baselines,
- optionally knn / knn-hybrid-β with disliked games (`<base>-neg-<mode>[-λ]`, recommend.DISLIKE_MODES), compared pairwise
  with their base on the users who have dislikes; `hiddenDislikes` also hides disliked games and checks they rank low.

With `center`, every vector strategy works on mean-centered vectors (recommend.center).

Filtered evaluation: only games that have a vector in Milvus take part (the system can't recommend anything else).

Niche discovery: the same hits counted only on hidden games outside the `nicheHead` most popular games
(`popular` scores ~0 there by construction), so finding liked games nobody plays becomes a number.
"""

import logging
import math
import random
import statistics
import time
from collections import Counter
from collections.abc import Callable
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.alias_generators import to_camel

from ml import recommend as rec
from ml.db import GameRepository, SteamUserData, SteamUserGame
from ml.tags import TagIndex
from ml.vectors import GameVectorStore

log = logging.getLogger(__name__)

# Popular games fetched once per run; enough to fill the top K even for users owning thousands of games
POPULAR_POOL = 20_000
# Hybrid candidates: gresym's nearest games (or kNN's candidates) plus the most popular games the user doesn't have
HYBRID_NEAREST = 200
HYBRID_POPULAR = 1000
# Popular games whose vectors are loaded once per run (the hybrid's popular candidates come from them)
HYBRID_POPULAR_LOADED = 3000
# Catalogue games sampled once per run to estimate where a hidden game ranks among all candidates
RANK_SAMPLE = 5000
# Most recommended games listed per strategy in the report
HUBS = 10
# Resamples of the paired bootstrap (dislike strategies vs their base)
BOOTSTRAP = 2000


class EvaluationError(Exception):
    """A run can't be done (no users, bad parameters); retrying won't help."""


class EvaluationParams(BaseModel):
    """Parameters of a run. JSON is camelCase (evaluation_runs.parameters is written by the backend)."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    ks: list[int] = Field(default=[5, 10, 20], min_length=1, max_length=10)
    test_size: int = Field(default=3, ge=1, le=20)  # hidden games per user
    min_train_games: int = Field(default=5, ge=1, le=100)  # liked games left after hiding, else the user is skipped
    min_playtime_minutes: int = Field(default=60, ge=0)  # an unreviewed game counts from this playtime on
    test_selection: Literal["top", "random-liked"] = "top"
    seed: int = 42
    hybrid_betas: list[float] = Field(default=[0.05, 0.1, 0.2, 0.4], max_length=10)  # one hybrid strategy per β
    center: bool = False  # mean-centered vectors for gresym, knn and the hybrids
    # Disliked games: one strategy per mode × λ (`exclude` has no λ) × base β (0 = knn, else knn-hybrid-β)
    dislike_modes: list[rec.DislikeMode] = Field(default=[], max_length=len(rec.DISLIKE_MODES))
    dislike_lambdas: list[float] = Field(default=[], max_length=10)
    dislike_betas: list[float] = Field(default=[0.0], max_length=10)
    hidden_dislikes: int = Field(default=0, ge=0, le=5)  # disliked games hidden per user (needs one more left in train)
    tags: bool = False  # tag-only strategies (loads ~3.2 M game_tags rows)
    niche_head: int = Field(default=1000, ge=1, le=POPULAR_POOL)  # the niche metrics ignore hidden games this popular

    @field_validator("ks")
    @classmethod
    def _check_ks(cls, ks: list[int]) -> list[int]:
        if any(k < 1 or k > 100 for k in ks):
            raise ValueError("every K must be between 1 and 100")
        return sorted(set(ks))

    @field_validator("hybrid_betas")
    @classmethod
    def _check_betas(cls, betas: list[float]) -> list[float]:
        if any(b <= 0 or b > 10 for b in betas):
            raise ValueError("every hybrid β must be in (0, 10]")
        return sorted(set(betas))

    @field_validator("dislike_lambdas")
    @classmethod
    def _check_lambdas(cls, lambdas: list[float]) -> list[float]:
        if any(lam <= 0 or lam > 10 for lam in lambdas):
            raise ValueError("every dislike λ must be in (0, 10]")
        return sorted(set(lambdas))

    @model_validator(mode="after")
    def _check_dislikes(self) -> EvaluationParams:
        self.dislike_modes = list(dict.fromkeys(self.dislike_modes))
        self.dislike_betas = sorted(set(self.dislike_betas))
        if any(beta != 0 and beta not in self.hybrid_betas for beta in self.dislike_betas):
            raise ValueError("every non-zero dislike β must also be a hybrid β (it's the paired baseline)")
        if any(mode != "exclude" for mode in self.dislike_modes) and not self.dislike_lambdas:
            raise ValueError("dislike modes other than exclude need dislike λs")
        return self

    def dislike_variants(self) -> list[tuple[rec.DislikeMode, float]]:
        """(mode, λ) pairs; `exclude` once with λ 0."""
        return [(mode, lam) for mode in self.dislike_modes
                for lam in ([0.0] if mode == "exclude" else self.dislike_lambdas)]

    def strategies(self) -> list[str]:
        return [
            "gresym", "knn",
            *(f"hybrid-{beta:g}" for beta in self.hybrid_betas),
            *(f"knn-hybrid-{beta:g}" for beta in self.hybrid_betas),
            *(dislike_strategy(beta, mode, lam) for beta in self.dislike_betas for mode, lam in self.dislike_variants()),
            *(["tags-top", "tags", *(f"tags-hybrid-{beta:g}" for beta in self.hybrid_betas)] if self.tags else []),
            "popular", "random",
        ]


def base_strategy(beta: float) -> str:
    return f"knn-hybrid-{beta:g}" if beta else "knn"


def dislike_strategy(beta: float, mode: str, lam: float) -> str:
    return f"{base_strategy(beta)}-neg-{mode}" + (f"-{lam:g}" if mode != "exclude" else "")


def _relevance(game: SteamUserGame) -> tuple[bool, int]:
    """Positive review first, then playtime."""
    return game.recommended is True, game.playtime_minutes


def split_user(
    games: list[SteamUserGame], params: EvaluationParams, rng: random.Random
) -> tuple[list[SteamUserGame], list[SteamUserGame]]:
    """(train, test). `games` are the user's usable games (with a vector). Test games are liked ones only."""
    liked = [g for g in games if g.recommended is not False]
    if params.test_selection == "top":
        test = sorted(liked, key=lambda g: (_relevance(g), -g.game_id), reverse=True)[: params.test_size]
    else:
        test = rng.sample(liked, min(params.test_size, len(liked)))
    test_ids = {g.game_id for g in test}
    return [g for g in games if g.game_id not in test_ids], test


def pick_hidden_dislikes(train: list[SteamUserGame], count: int, rng: random.Random) -> list[SteamUserGame]:
    """`count` disliked train games to hide, only if at least one more stays in train (else none)."""
    disliked = sorted((g for g in train if g.recommended is False), key=lambda g: g.game_id)
    return rng.sample(disliked, count) if count and len(disliked) > count else []


def train_ratings(train: list[SteamUserGame]) -> list[rec.Rating]:
    """Reviews give ±1 (recommend). Unreviewed games are likes scaled by playtime: percentile within the user → 6..10."""
    ratings = [rec.Rating(g.game_id, recommend=g.recommended) for g in train if g.recommended is not None]
    unreviewed = sorted((g for g in train if g.recommended is None), key=lambda g: (g.playtime_minutes, g.game_id))
    for rank, game in enumerate(unreviewed):
        share = rank / (len(unreviewed) - 1) if len(unreviewed) > 1 else 1.0
        ratings.append(rec.Rating(game.game_id, rating=round(6 + 4 * share)))
    return ratings


def hit_ranks(recommended: list[int], test_ids: set[int]) -> list[int]:
    """1-based positions of hidden games in the recommendation list."""
    return [position for position, game_id in enumerate(recommended, start=1) if game_id in test_ids]


def metrics_at(ranks: list[int], test_count: int, k: int) -> dict[str, float]:
    hits = [r for r in ranks if r <= k]
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, test_count)))
    return {
        "precision": len(hits) / k,
        "recall": len(hits) / test_count,
        "hitRate": 1.0 if hits else 0.0,
        "ndcg": sum(1 / math.log2(r + 1) for r in hits) / ideal,
    }


def hits_spread(user_ranks: list[list[int]], names: list[str], k: int) -> dict:
    """Hits in the top k per user: the fewest and the most, who got them, and how many users got each number of hits."""
    hits = [sum(1 for r in ranks if r <= k) for ranks in user_ranks]
    if not hits:
        return {"k": k, "min": None, "max": None, "minUsers": [], "maxUsers": [], "distribution": {}}
    low, high = min(hits), max(hits)
    return {
        "k": k, "min": low, "max": high,
        "minUsers": [name for name, n in zip(names, hits) if n == low],
        "maxUsers": [name for name, n in zip(names, hits) if n == high],
        "distribution": {str(n): c for n, c in sorted(Counter(hits).items())},
    }


def rank_percentiles(test_scores: np.ndarray, sample_scores: np.ndarray) -> list[float]:
    """Share of the sampled candidates that score above each hidden game (ties count half): 0 = top, 0.5 = random."""
    n = len(sample_scores)
    if n == 0:
        return []
    return [float(((sample_scores > t).sum() + 0.5 * (sample_scores == t).sum()) / n) for t in test_scores]


def _first_free(candidates: list[int], excluded: set[int], limit: int) -> list[int]:
    result = []
    for game_id in candidates:
        if game_id not in excluded:
            result.append(game_id)
            if len(result) == limit:
                break
    return result


def _random_games(all_ids: list[int], excluded: set[int], limit: int, rng: random.Random) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    while len(result) < limit and len(seen) < len(all_ids):
        game_id = rng.choice(all_ids)
        if game_id not in seen:
            seen.add(game_id)
            if game_id not in excluded:
                result.append(game_id)
    return result


def _matrix(ids: list[int], vectors: dict[int, list[float]]) -> np.ndarray:
    return np.asarray([vectors[i] for i in ids], dtype=np.float32).reshape(len(ids), -1)


def evaluate(
    params: EvaluationParams,
    users: list[SteamUserData],
    store: GameVectorStore,
    popular_ids: list[int],
    all_game_ids: list[int],
    recommendation_counts: dict[int, int] | None = None,
    progress: Callable[[int, int], None] | None = None,
    tag_index: TagIndex | None = None,
) -> dict:
    """Runs leave-k-out over all users. Returns the JSON-ready report stored in evaluation_runs.results.

    The tag strategies need `tag_index` (built over `all_game_ids`) when `params.tags` is set.
    """
    if params.tags and tag_index is None:
        raise EvaluationError("The tag strategies need the game tags")
    started = time.monotonic()
    max_k = params.ks[-1]
    strategies = params.strategies()
    counts = recommendation_counts or {}
    max_count = max(counts.values(), default=1)

    def pop(ids: list[int]) -> np.ndarray:
        return rec.popularity(np.asarray([counts.get(i, 0) for i in ids], dtype=np.float64), max_count)

    # Loaded once per run: the sample for rank percentiles (and the catalogue mean) and the hybrids' popular candidates
    sample_rng = random.Random(f"{params.seed}:rank-sample")
    sample_vectors = store.get_vectors(sample_rng.sample(all_game_ids, min(RANK_SAMPLE, len(all_game_ids))))
    sample_ids = np.asarray(list(sample_vectors), dtype=np.int64)
    sample_raw = _matrix(list(sample_vectors), sample_vectors)
    # The mean of a 5000-game random sample is a close estimate of the catalogue mean
    mean = sample_raw.mean(axis=0) if params.center and len(sample_raw) else None

    def space(ids: list[int], vectors: dict[int, list[float]]) -> np.ndarray:
        """The vectors the strategies compare: as stored, or centered."""
        return rec.center(_matrix(ids, vectors), mean)

    sample_matrix = rec.center(sample_raw, mean)
    sample_pop = pop(sample_ids.tolist())
    popular_vectors = store.get_vectors(popular_ids[:HYBRID_POPULAR_LOADED])
    popular_loaded = [i for i in popular_ids[:HYBRID_POPULAR_LOADED] if i in popular_vectors]
    popular_row = {game_id: row for row, game_id in enumerate(popular_loaded)}
    popular_matrix = space(popular_loaded, popular_vectors)
    # Popularity of every catalogue game in tag_index order, and the rank sample's rows there
    catalogue_pop = pop(tag_index.ids.tolist()) if params.tags else None
    sample_tag_rows = tag_index.rows(sample_ids) if params.tags else None

    def add_tag_strategies(ratings: list[rec.Rating], excluded: set[int], probe_ids: list[int], free: np.ndarray,
                           lists: dict[str, list[int]], probe_percentiles: dict[str, list[float]],
                           probe_pop: np.ndarray, candidates_pop: np.ndarray) -> None:
        """tags-top, tags and tags-hybrid-β, each over the whole catalogue (owned games blocked)."""
        liked = [(r.game_id, w) for r in ratings if (w := rec.rating_weight(r)) > 0]
        profile = tag_index.profile([game_id for game_id, _ in liked], [w for _, w in liked])
        blocked = tag_index.rows(sorted(excluded))
        probe_rows, sample_rows = tag_index.rows(probe_ids), sample_tag_rows[free]

        def add(name: str, scores: np.ndarray, probe_scores: np.ndarray, sample_scores: np.ndarray) -> None:
            scores = scores.copy()
            scores[blocked] = -np.inf
            top = np.argpartition(-scores, max_k)[:max_k]
            top = top[np.lexsort((tag_index.ids[top], -scores[top]))]
            lists[name] = [int(tag_index.ids[i]) for i in top if np.isfinite(scores[i])]
            probe_percentiles[name] = rank_percentiles(probe_scores, sample_scores)

        # The literal approach: filter by the favourite tags, sort by popularity (the count dominates, popularity < 1 breaks ties)
        matches = tag_index.match_counts(tag_index.top_tags(profile)) + catalogue_pop
        add("tags-top", matches, matches[probe_rows], matches[sample_rows])
        cosine = tag_index.scores(profile)
        add("tags", cosine, cosine[probe_rows], cosine[sample_rows])
        for beta in params.hybrid_betas:
            add(f"tags-hybrid-{beta:g}", cosine + beta * catalogue_pop,
                cosine[probe_rows] + beta * probe_pop, cosine[sample_rows] + beta * candidates_pop)

    skipped: Counter[str] = Counter()
    filtered: Counter[str] = Counter()
    per_user: list[dict] = []
    # strategy → list of rank lists, one per evaluated user
    ranks: dict[str, list[list[int]]] = {s: [] for s in strategies}
    percentiles: dict[str, list[float]] = {s: [] for s in strategies if s != "random"}
    # Per evaluated user, for the paired comparison of the dislike strategies: mean rank percentile, train dislikes
    user_mean_percentiles: dict[str, list[float]] = {s: [] for s in percentiles}
    user_dislikes: list[int] = []
    # Hidden dislikes (hiddenDislikes > 0): their rank percentiles and, per user who has them, their hit ranks
    dislike_percentiles: dict[str, list[float]] = {s: [] for s in percentiles}
    dislike_ranks: dict[str, list[list[int]]] = {s: [] for s in strategies}
    test_counts: list[int] = []
    # Niche metrics: only hidden games outside the popular head, only users who have such a game hidden
    head = set(popular_ids[: params.niche_head])
    niche_ranks: dict[str, list[list[int]]] = {s: [] for s in strategies}
    niche_counts: list[int] = []
    niche_names: list[str] = []
    niche_percentiles: dict[str, list[float]] = {s: [] for s in percentiles}
    recommended_counts: dict[str, Counter[int]] = {s: Counter() for s in strategies}

    for done, user in enumerate(users, start=1):
        if progress is not None:
            progress(done, len(users))
        # The seed mixes the user id in, so results don't depend on the order or number of users
        rng = random.Random(f"{params.seed}:{user.id}")
        usable = [g for g in user.games if g.recommended is not None or g.playtime_minutes >= params.min_playtime_minutes]
        vectors = store.get_vectors([g.game_id for g in usable])
        with_vector = [g for g in usable if g.game_id in vectors]
        filtered["withoutVector"] += len(usable) - len(with_vector)

        liked = sum(1 for g in with_vector if g.recommended is not False)
        if liked < params.test_size + params.min_train_games:
            skipped["tooFewLikedGames"] += 1
            continue

        train, test = split_user(with_vector, params, rng)
        # A separate generator, so hiding dislikes doesn't change the other random choices
        hidden = pick_hidden_dislikes(train, params.hidden_dislikes, random.Random(f"{params.seed}:{user.id}:dislikes"))
        hidden_ids = sorted(g.game_id for g in hidden)
        train = [g for g in train if g.game_id not in set(hidden_ids)]
        ratings = train_ratings(train)
        user_ids = [g.game_id for g in with_vector]
        user_vectors = dict(zip(user_ids, space(user_ids, vectors)))
        # The averaged profile (recommend.build_profile), from centered vectors with `center`
        profile, _ = rec.build_profile(ratings, user_vectors)
        if profile is None:
            skipped["noProfile"] += 1
            continue
        test_ids = sorted(g.game_id for g in test)
        # Everything else the user has (low playtime, no vector, ...) must not count as a miss either
        owned = {g.game_id for g in user.games}
        excluded = owned - set(test_ids) - set(hidden_ids)
        excluded_list = sorted(excluded)

        def popular_extra(taken: list[int]) -> tuple[list[int], np.ndarray]:
            """The most popular free games not in `taken`, with their vectors (the hybrids' second half of the pool)."""
            ids = _first_free(popular_loaded, excluded | set(taken), HYBRID_POPULAR)
            return ids, popular_matrix[[popular_row[i] for i in ids]].reshape(len(ids), -1)

        def fetch(ids: list[int]) -> tuple[list[int], np.ndarray]:
            fetched = store.get_vectors(ids)
            ids = [i for i in ids if i in fetched]
            return ids, space(ids, fetched)

        lists: dict[str, list[int]] = {}

        def rank_hybrids(prefix: str, ids: list[int], scores: np.ndarray, own: str, own_count: int) -> None:
            """`own` = the plain strategy (top by score over its first `own_count` candidates, i.e. without the popular
            extras), then one list per β with popularity."""
            lists[own] = [ids[i] for i in np.argsort(-scores[:own_count], kind="stable")[:max_k]]
            pool_pop = pop(ids)
            for beta in params.hybrid_betas:
                order = np.argsort(-(scores + beta * pool_pop), kind="stable")[:max_k]
                lists[f"{prefix}-{beta:g}"] = [ids[i] for i in order]

        # gresym: Milvus ranks by profile·(v − mean) up to a per-game norm, so the pool is re-scored exactly
        nearest_ids, nearest_matrix = fetch(
            [game_id for game_id, _ in store.search(profile.tolist(), max(max_k, HYBRID_NEAREST), excluded_list)]
        )
        extra_ids, extra_matrix = popular_extra(nearest_ids)
        rank_hybrids("hybrid", nearest_ids + extra_ids, np.concatenate([nearest_matrix, extra_matrix]) @ profile,
                     "gresym", len(nearest_ids))

        liked_vectors, weights = rec.liked_matrix(ratings, user_vectors)
        knn_ids, knn_matrix = fetch(rec.knn_candidates(store, liked_vectors, excluded_list))
        extra_ids, extra_matrix = popular_extra(knn_ids)
        knn_pool_ids, knn_pool = knn_ids + extra_ids, np.concatenate([knn_matrix, extra_matrix])
        knn_pool_scores = rec.knn_scores(knn_pool, liked_vectors, weights)
        rank_hybrids("knn-hybrid", knn_pool_ids, knn_pool_scores, "knn", len(knn_ids))

        lists["popular"] = _first_free(popular_ids, excluded, max_k)
        lists["random"] = _random_games(all_game_ids, excluded, max_k, rng)

        # Rank percentiles: the strategy's score of each hidden game vs the sampled games the user doesn't have.
        # The probes are the hidden liked games, then the hidden disliked ones
        free = ~np.isin(sample_ids, list(owned))
        candidates, candidates_pop = sample_matrix[free], sample_pop[free]
        probe_ids = test_ids + hidden_ids
        probe_matrix, probe_pop = space(probe_ids, vectors), pop(probe_ids)
        sims_probe, sims_sample = probe_matrix @ profile, candidates @ profile
        knn_probe = rec.knn_scores(probe_matrix, liked_vectors, weights)
        knn_sample = rec.knn_scores(candidates, liked_vectors, weights)
        probe_percentiles = {
            "gresym": rank_percentiles(sims_probe, sims_sample),
            "knn": rank_percentiles(knn_probe, knn_sample),
            **{
                f"hybrid-{beta:g}": rank_percentiles(sims_probe + beta * probe_pop, sims_sample + beta * candidates_pop)
                for beta in params.hybrid_betas
            },
            **{
                f"knn-hybrid-{beta:g}": rank_percentiles(knn_probe + beta * probe_pop, knn_sample + beta * candidates_pop)
                for beta in params.hybrid_betas
            },
            "popular": rank_percentiles(probe_pop, candidates_pop),
        }

        # Dislike strategies: kNN scores changed by the disliked games, ranked like knn / knn-hybrid-β
        disliked, dweights = rec.disliked_matrix(ratings, user_vectors)

        def add_dislike_strategies(mode: str, lam: float, ids: list[int], pool_scores: np.ndarray, own_count: int,
                                   probe_scores: np.ndarray, sample_scores: np.ndarray) -> None:
            pool_pop = pop(ids)
            for beta in params.dislike_betas:
                name = dislike_strategy(beta, mode, lam)
                scores = pool_scores[:own_count] if beta == 0 else pool_scores + beta * pool_pop
                order = np.argsort(-scores, kind="stable")
                lists[name] = [ids[i] for i in order if np.isfinite(scores[i])][:max_k]
                probe_percentiles[name] = rank_percentiles(probe_scores + beta * probe_pop,
                                                           sample_scores + beta * candidates_pop)

        penalties: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for mode, lam in params.dislike_variants():
            if mode == "rocchio":
                shifted = rec.rocchio_shift(liked_vectors, disliked, dweights, lam)
                if shifted is liked_vectors:  # no dislikes: exactly knn
                    add_dislike_strategies(mode, lam, knn_pool_ids, knn_pool_scores, len(knn_ids), knn_probe, knn_sample)
                    continue
                shifted_ids, shifted_matrix = fetch(rec.knn_candidates(store, shifted, excluded_list))
                extra_ids, extra_matrix = popular_extra(shifted_ids)
                add_dislike_strategies(
                    mode, lam, shifted_ids + extra_ids,
                    rec.knn_scores(np.concatenate([shifted_matrix, extra_matrix]), shifted, weights), len(shifted_ids),
                    rec.knn_scores(probe_matrix, shifted, weights), rec.knn_scores(candidates, shifted, weights),
                )
            elif mode == "exclude":
                banned = list(rec.dislike_neighbours(store, disliked, excluded_list))

                def ban(ids: list[int] | np.ndarray) -> np.ndarray:
                    return np.where(np.isin(ids, banned), -np.inf, 0.0)

                add_dislike_strategies(mode, lam, knn_pool_ids, knn_pool_scores + ban(knn_pool_ids), len(knn_ids),
                                       knn_probe + ban(probe_ids), knn_sample + ban(sample_ids[free]))
            else:
                if mode not in penalties:
                    penalties[mode] = tuple(  # type: ignore[assignment]
                        rec.dislike_penalty(matrix, disliked, dweights, mode, liked_vectors, weights)
                        for matrix in (knn_pool, probe_matrix, candidates)
                    )
                pen_pool, pen_probe, pen_sample = penalties[mode]
                add_dislike_strategies(mode, lam, knn_pool_ids, knn_pool_scores - lam * pen_pool, len(knn_ids),
                                       knn_probe - lam * pen_probe, knn_sample - lam * pen_sample)

        if params.tags:
            add_tag_strategies(ratings, excluded, probe_ids, free, lists, probe_percentiles, probe_pop, candidates_pop)

        test_count = len(test_ids)
        user_percentiles = {s: values[:test_count] for s, values in probe_percentiles.items()}
        hidden_percentiles = {s: values[test_count:] for s, values in probe_percentiles.items()}

        test_set = set(test_ids)
        test_counts.append(test_count)
        niche_test = [game_id for game_id in test_ids if game_id not in head]
        if niche_test:
            niche_counts.append(len(niche_test))
            niche_names.append(user.name)
            for strategy in strategies:
                niche_ranks[strategy].append(hit_ranks(lists[strategy], set(niche_test)))
            for strategy, values in user_percentiles.items():
                niche_percentiles[strategy].extend(v for game_id, v in zip(test_ids, values) if game_id not in head)
        user_dislikes.append(len(disliked))
        user_ranks = {}
        for strategy in strategies:
            user_ranks[strategy] = hit_ranks(lists[strategy], test_set)
            ranks[strategy].append(user_ranks[strategy])
            recommended_counts[strategy].update(lists[strategy])
            if hidden_ids:
                dislike_ranks[strategy].append(hit_ranks(lists[strategy], set(hidden_ids)))
        for strategy, values in user_percentiles.items():
            percentiles[strategy].extend(values)
            user_mean_percentiles[strategy].append(statistics.fmean(values))
            dislike_percentiles[strategy].extend(hidden_percentiles[strategy])
        per_user.append({
            "name": user.name,
            "games": len(user.games),
            "train": len(train),
            "dislikes": len(disliked),
            "test": test_ids,
            "nicheTest": niche_test,
            "ranks": user_ranks,
            "rankPercentiles": {s: [round(v, 4) for v in values] for s, values in user_percentiles.items()},
            "recommended": {s: games for s, games in lists.items() if s != "random"},
            **({
                "hiddenDislikes": hidden_ids,
                "dislikeRanks": {s: dislike_ranks[s][-1] for s in strategies},
                "dislikeRankPercentiles": {s: [round(v, 4) for v in values] for s, values in hidden_percentiles.items()},
            } if hidden_ids else {}),
        })

    evaluated = len(test_counts)
    summaries: dict[str, dict] = {}
    for strategy in strategies:
        summary: dict = {}
        for k in params.ks:
            values = [metrics_at(r, n, k) for r, n in zip(ranks[strategy], test_counts)]
            summary[str(k)] = {
                name: (sum(v[name] for v in values) / evaluated if evaluated else 0.0)
                for name in ("precision", "recall", "hitRate", "ndcg")
            }
        summary["mrr"] = sum(1 / r[0] for r in ranks[strategy] if r) / evaluated if evaluated else 0.0
        summary["hits"] = hits_spread(ranks[strategy], [u["name"] for u in per_user], max_k)
        # Share of the catalogue that appeared in anyone's top max K
        summary["coverage"] = len(recommended_counts[strategy]) / len(all_game_ids) if all_game_ids else 0.0
        # How mainstream the recommendations are: median Steam recommendations count over all recommended slots
        slots = [counts.get(game_id, 0) for game_id in recommended_counts[strategy].elements()]
        summary["medianRecommendationsCount"] = statistics.median(slots) if slots else 0
        if strategy in percentiles:
            values = percentiles[strategy]
            summary["rankPercentile"] = {
                "median": statistics.median(values) if values else None,
                "mean": statistics.fmean(values) if values else None,
            }
        # The same metrics over the niche hidden games only (a hit on a head game doesn't count)
        niche_users = len(niche_counts)
        niche: dict = {"users": niche_users}
        for k in params.ks:
            values = [metrics_at(r, n, k) for r, n in zip(niche_ranks[strategy], niche_counts)]
            niche[str(k)] = {name: (sum(v[name] for v in values) / niche_users if niche_users else 0.0)
                             for name in ("recall", "hitRate")}
        niche["mrr"] = sum(1 / r[0] for r in niche_ranks[strategy] if r) / niche_users if niche_users else 0.0
        niche["hits"] = hits_spread(niche_ranks[strategy], niche_names, max_k)
        if strategy in niche_percentiles and (values := niche_percentiles[strategy]):
            niche["rankPercentile"] = {"median": statistics.median(values), "mean": statistics.fmean(values)}
        summary["niche"] = niche
        if params.hidden_dislikes:
            # Higher rank percentile and lower hit rate = the strategy keeps the hidden disliked games away
            with_hidden = dislike_ranks[strategy]
            summary["dislike"] = {
                "users": len(with_hidden),
                "hitRate": {str(k): (sum(1 for r in with_hidden if r and r[0] <= k) / len(with_hidden)
                                     if with_hidden else 0.0) for k in params.ks},
            }
            if strategy in dislike_percentiles and (values := dislike_percentiles[strategy]):
                summary["dislike"]["rankPercentile"] = {"median": statistics.median(values), "mean": statistics.fmean(values)}
        summaries[strategy] = summary

    # Dislike strategies vs their base, paired per user, only on users with disliked games in train
    # (for everyone else the lists are identical by construction)
    with_dislikes = [i for i, n in enumerate(user_dislikes) if n > 0]
    paired = {}
    for beta in params.dislike_betas:
        base = base_strategy(beta)
        for mode, lam in params.dislike_variants():
            strategy = dislike_strategy(beta, mode, lam)
            paired[strategy] = {"base": base, **paired_comparison(
                [ranks[base][i] for i in with_dislikes], [ranks[strategy][i] for i in with_dislikes],
                [user_mean_percentiles[base][i] for i in with_dislikes],
                [user_mean_percentiles[strategy][i] for i in with_dislikes],
                max_k, np.random.default_rng(params.seed),
            )}

    return {
        "params": params.model_dump(by_alias=True),
        "users": {"total": len(users), "evaluated": evaluated, "skipped": dict(skipped)},
        "filtered": dict(filtered),
        "rankSample": len(sample_ids),
        "niche": {"head": params.niche_head, "users": len(niche_counts), "hiddenGames": sum(niche_counts),
                  "hiddenGamesTotal": sum(test_counts)},
        "centered": mean is not None,
        "strategies": summaries,
        "paired": paired,
        # The games most often in a top max K: a few games recommended to many users = "hubs" of the vector space
        "hubs": {
            strategy: [{"gameId": game_id, "users": n} for game_id, n in recommended_counts[strategy].most_common(HUBS)]
            for strategy in strategies if strategy != "random"
        },
        "perUser": per_user,
        "durationSeconds": round(time.monotonic() - started, 2),
    }


def paired_comparison(
    base_ranks: list[list[int]], ranks: list[list[int]], base_percentiles: list[float], percentiles: list[float],
    k: int, rng: np.random.Generator,
) -> dict:
    """Per-user differences strategy − base (HR@k, reciprocal rank, mean rank percentile) with paired bootstrap 95 % CIs.

    A negative rank-percentile delta is better. `won`/`lost` = users with more/fewer hits in the top k.
    """
    n = len(ranks)
    if n == 0:
        return {"users": 0}

    def hits(r: list[int]) -> int:
        return sum(1 for position in r if position <= k)

    columns = {
        "hitRate": (np.array([min(hits(r), 1) for r in base_ranks], dtype=float),
                    np.array([min(hits(r), 1) for r in ranks], dtype=float)),
        "mrr": (np.array([1 / r[0] if r else 0.0 for r in base_ranks]), np.array([1 / r[0] if r else 0.0 for r in ranks])),
        "rankPercentile": (np.asarray(base_percentiles), np.asarray(percentiles)),
    }
    samples = rng.integers(0, n, size=(BOOTSTRAP, n))
    result: dict = {"users": n, "k": k}
    for name, (base, value) in columns.items():
        means = (value - base)[samples].mean(axis=1)
        result[name] = {
            "base": float(base.mean()), "value": float(value.mean()), "delta": float((value - base).mean()),
            "ci": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        }
    result["won"] = sum(1 for b, r in zip(base_ranks, ranks) if hits(r) > hits(b))
    result["lost"] = sum(1 for b, r in zip(base_ranks, ranks) if hits(r) < hits(b))
    return result


def run_evaluation(
    params: EvaluationParams,
    store: GameVectorStore,
    games: GameRepository,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Loads the users and the baselines' inputs from MariaDB, then evaluates."""
    users = games.get_steam_users()
    if not users:
        raise EvaluationError("No Steam users in the database: collect and import them first (docs/evaluation.md)")
    log.info("Evaluating %d Steam users with %s", len(users), params.model_dump(by_alias=True))
    all_game_ids = games.get_game_ids()
    tag_index = TagIndex(*games.get_game_tags(), all_game_ids) if params.tags else None
    report = evaluate(
        params, users, store, games.get_popular_game_ids(POPULAR_POOL), all_game_ids,
        games.get_recommendation_counts(), progress, tag_index,
    )
    titles = games.get_titles(sorted({hub["gameId"] for hubs in report["hubs"].values() for hub in hubs}))
    for hubs in report["hubs"].values():
        for hub in hubs:
            hub["title"] = titles.get(hub["gameId"])
    return report


def _hits_lines(strategies: dict, spread: Callable[[dict], dict], label: str) -> list[str]:
    """Per strategy: fewest/most hits per user, the users with the most, and the distribution (hits: users)."""
    first = next(iter(strategies.values()), None)
    if first is None or spread(first)["min"] is None:
        return []
    lines = ["", f"Per-user {label} in the top {spread(first)['k']}: min (users) / max (users with it), distribution hits:users"]
    for strategy, summary in strategies.items():
        h = spread(summary)
        best = ", ".join(h["maxUsers"][:5]) + (f" +{len(h['maxUsers']) - 5}" if len(h["maxUsers"]) > 5 else "")
        dist = " ".join(f"{n}:{c}" for n, c in h["distribution"].items())
        lines.append(f"  {strategy:<36} {h['min']} ({len(h['minUsers'])}) / {h['max']} ({best})  [{dist}]")
    return lines


def format_report(report: dict) -> str:
    """Plain-text table for `ml evaluate`."""
    users = report["users"]
    evaluated = users["evaluated"]
    lines = [
        f"Users: {evaluated} evaluated of {users['total']}, skipped {users['skipped'] or 0}; "
        f"games without a vector: {report['filtered'].get('withoutVector', 0)}; {report['durationSeconds']} s",
        "",
        f"{'strategy':<36} {'K':>3} {'P@K':>7} {'R@K':>7} {'HR@K':>7} {'nDCG@K':>7}",
    ]
    for strategy, summary in report["strategies"].items():
        for k in report["params"]["ks"]:
            m = summary[str(k)]
            lines.append(
                f"{strategy:<36} {k:>3} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['hitRate']:>7.4f} {m['ndcg']:>7.4f}"
            )
        extra = f"{strategy:<36} MRR {summary['mrr']:.4f}, coverage {summary['coverage']:.4%}, " \
                f"median recommendations count {summary['medianRecommendationsCount']:g}"
        if (percentile := summary.get("rankPercentile")) and percentile["median"] is not None:
            extra += f", rank percentile median {percentile['median']:.2%} / mean {percentile['mean']:.2%}"
        lines.append(extra)
    lines += _hits_lines(report["strategies"], lambda summary: summary["hits"], "hits")
    if (niche := report.get("niche")) and niche["users"]:
        ks = report["params"]["ks"]
        lines += ["", (f"Niche discovery: hits on hidden games outside the top {niche['head']} popular only; "
                       f"{niche['hiddenGames']} of {niche['hiddenGamesTotal']} hidden games, {niche['users']} users have one"),
                  f"{'strategy':<36} " + " ".join(f"{'HR@' + str(k):>7} {'R@' + str(k):>7}" for k in ks)
                  + f" {'MRR':>7} {'rank pct med/mean':>18}"]
        for strategy, summary in report["strategies"].items():
            n = summary["niche"]
            pct = n.get("rankPercentile")
            lines.append(f"{strategy:<36} " + " ".join(f"{n[str(k)]['hitRate']:>7.4f} {n[str(k)]['recall']:>7.4f}" for k in ks)
                         + f" {n['mrr']:>7.4f} " + (f"{pct['median']:>8.2%} / {pct['mean']:.2%}" if pct else f"{'–':>18}"))
        lines += _hits_lines(report["strategies"], lambda summary: summary["niche"]["hits"], "niche hits")
    if paired := report.get("paired"):
        first = next(iter(paired.values()))
        lines += ["", (f"Dislike strategies vs their base, users with dislikes in train ({first.get('users', 0)}), "
                       "Δ = strategy − base [95 % CI]; rank percentile: lower is better"),
                  f"{'strategy':<36} {'ΔHR@' + str(first.get('k', '')):>24} {'ΔMRR':>24} {'Δrank pct':>26} {'won/lost':>9}"]
        for strategy, p in paired.items():
            if not p.get("users"):
                continue
            cells = [f"{p[m]['delta']:+.3f} [{p[m]['ci'][0]:+.3f}, {p[m]['ci'][1]:+.3f}]" for m in ("hitRate", "mrr")]
            pct = p["rankPercentile"]
            cells.append(f"{pct['delta']:+.2%} [{pct['ci'][0]:+.2%}, {pct['ci'][1]:+.2%}]")
            lines.append(f"{strategy:<36} {cells[0]:>24} {cells[1]:>24} {cells[2]:>26} {p['won']:>4}/{p['lost']:<4}")
    if report["params"].get("hiddenDislikes"):
        lines += ["", "Hidden disliked games: rank percentile (higher = kept away better), hit rate (lower = better)"]
        for strategy, summary in report["strategies"].items():
            d = summary.get("dislike")
            if not d or not d["users"]:
                continue
            pct = d.get("rankPercentile")
            pct_text = f"median {pct['median']:.2%} / mean {pct['mean']:.2%}" if pct else "–"
            hr = ", ".join(f"HR@{k} {v:.3f}" for k, v in d["hitRate"].items())
            lines.append(f"  {strategy:<36} users {d['users']}, {pct_text}, {hr}")
    lines += ["", f"Most recommended games (users of {evaluated} with the game in their top {report['params']['ks'][-1]}):"]
    for strategy, hubs in report["hubs"].items():
        listed = ", ".join(f"{hub.get('title') or hub['gameId']} ({hub['users']})" for hub in hubs[:5])
        lines.append(f"  {strategy:<36} {listed}")
    return "\n".join(lines)
