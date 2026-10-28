"""Offline evaluation of recommendations on collected Steam users (leave-k-out, see docs/evaluation.md).

For every user some liked games are hidden (test), recommendations are built from the rest (train), and the
metrics measure how many hidden games come back in the top K. Strategies compared:
- gresym: the averaged taste profile searched in Milvus (POST /recommendations until 2026-10-27),
- knn: item-kNN over single liked games (recommend.knn_candidates + knn_scores),
- hybrid-β: gresym's similarity + β·popularity, re-ranking gresym's nearest games together with the most popular ones,
- knn-hybrid-β: the same with the item-kNN score (with `center`, what POST /recommendations does: recommend.recommend),
- popular (the API's fallback) and random games as baselines.

With `center`, every vector strategy works on mean-centered vectors (recommend.center).

Filtered evaluation: only games that have a vector in Milvus take part (the system can't recommend anything else).
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
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel

from ml import recommend as rec
from ml.db import GameRepository, SteamUserData, SteamUserGame
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

    def strategies(self) -> list[str]:
        return [
            "gresym", "knn",
            *(f"hybrid-{beta:g}" for beta in self.hybrid_betas),
            *(f"knn-hybrid-{beta:g}" for beta in self.hybrid_betas),
            "popular", "random",
        ]


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
) -> dict:
    """Runs leave-k-out over all users. Returns the JSON-ready report stored in evaluation_runs.results."""
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

    skipped: Counter[str] = Counter()
    filtered: Counter[str] = Counter()
    per_user: list[dict] = []
    # strategy → list of rank lists, one per evaluated user
    ranks: dict[str, list[list[int]]] = {s: [] for s in strategies}
    percentiles: dict[str, list[float]] = {s: [] for s in strategies if s != "random"}
    test_counts: list[int] = []
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
        excluded = owned - set(test_ids)
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
        rank_hybrids("knn-hybrid", knn_ids + extra_ids,
                     rec.knn_scores(np.concatenate([knn_matrix, extra_matrix]), liked_vectors, weights), "knn", len(knn_ids))

        lists["popular"] = _first_free(popular_ids, excluded, max_k)
        lists["random"] = _random_games(all_game_ids, excluded, max_k, rng)

        # Rank percentiles: the strategy's score of each hidden game vs the sampled games the user doesn't have
        free = ~np.isin(sample_ids, list(owned))
        candidates, candidates_pop = sample_matrix[free], sample_pop[free]
        test_matrix, test_pop = space(test_ids, vectors), pop(test_ids)
        sims_test, sims_sample = test_matrix @ profile, candidates @ profile
        knn_test = rec.knn_scores(test_matrix, liked_vectors, weights)
        knn_sample = rec.knn_scores(candidates, liked_vectors, weights)
        user_percentiles = {
            "gresym": rank_percentiles(sims_test, sims_sample),
            "knn": rank_percentiles(knn_test, knn_sample),
            **{
                f"hybrid-{beta:g}": rank_percentiles(sims_test + beta * test_pop, sims_sample + beta * candidates_pop)
                for beta in params.hybrid_betas
            },
            **{
                f"knn-hybrid-{beta:g}": rank_percentiles(knn_test + beta * test_pop, knn_sample + beta * candidates_pop)
                for beta in params.hybrid_betas
            },
            "popular": rank_percentiles(test_pop, candidates_pop),
        }

        test_set = set(test_ids)
        test_counts.append(len(test_ids))
        user_ranks = {}
        for strategy in strategies:
            user_ranks[strategy] = hit_ranks(lists[strategy], test_set)
            ranks[strategy].append(user_ranks[strategy])
            recommended_counts[strategy].update(lists[strategy])
        for strategy, values in user_percentiles.items():
            percentiles[strategy].extend(values)
        per_user.append({
            "name": user.name,
            "games": len(user.games),
            "train": len(train),
            "test": test_ids,
            "ranks": user_ranks,
            "rankPercentiles": {s: [round(v, 4) for v in values] for s, values in user_percentiles.items()},
            "recommended": {s: games for s, games in lists.items() if s != "random"},
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
        summaries[strategy] = summary

    return {
        "params": params.model_dump(by_alias=True),
        "users": {"total": len(users), "evaluated": evaluated, "skipped": dict(skipped)},
        "filtered": dict(filtered),
        "rankSample": len(sample_ids),
        "centered": mean is not None,
        "strategies": summaries,
        # The games most often in a top max K: a few games recommended to many users = "hubs" of the vector space
        "hubs": {
            strategy: [{"gameId": game_id, "users": n} for game_id, n in recommended_counts[strategy].most_common(HUBS)]
            for strategy in strategies if strategy != "random"
        },
        "perUser": per_user,
        "durationSeconds": round(time.monotonic() - started, 2),
    }


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
    report = evaluate(
        params, users, store, games.get_popular_game_ids(POPULAR_POOL), games.get_game_ids(),
        games.get_recommendation_counts(), progress,
    )
    titles = games.get_titles(sorted({hub["gameId"] for hubs in report["hubs"].values() for hub in hubs}))
    for hubs in report["hubs"].values():
        for hub in hubs:
            hub["title"] = titles.get(hub["gameId"])
    return report


def format_report(report: dict) -> str:
    """Plain-text table for `ml evaluate`."""
    users = report["users"]
    evaluated = users["evaluated"]
    lines = [
        f"Users: {evaluated} evaluated of {users['total']}, skipped {users['skipped'] or 0}; "
        f"games without a vector: {report['filtered'].get('withoutVector', 0)}; {report['durationSeconds']} s",
        "",
        f"{'strategy':<17} {'K':>3} {'P@K':>7} {'R@K':>7} {'HR@K':>7} {'nDCG@K':>7}",
    ]
    for strategy, summary in report["strategies"].items():
        for k in report["params"]["ks"]:
            m = summary[str(k)]
            lines.append(
                f"{strategy:<17} {k:>3} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['hitRate']:>7.4f} {m['ndcg']:>7.4f}"
            )
        extra = f"{strategy:<17} MRR {summary['mrr']:.4f}, coverage {summary['coverage']:.4%}, " \
                f"median recommendations count {summary['medianRecommendationsCount']:g}"
        if (percentile := summary.get("rankPercentile")) and percentile["median"] is not None:
            extra += f", rank percentile median {percentile['median']:.2%} / mean {percentile['mean']:.2%}"
        lines.append(extra)
    lines += ["", f"Most recommended games (users of {evaluated} with the game in their top {report['params']['ks'][-1]}):"]
    for strategy, hubs in report["hubs"].items():
        listed = ", ".join(f"{hub.get('title') or hub['gameId']} ({hub['users']})" for hub in hubs[:5])
        lines.append(f"  {strategy:<17} {listed}")
    return "\n".join(lines)
