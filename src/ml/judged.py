"""`ml judge`: LLM-judged evaluation of the lists in an `ml evaluate --json` report (docs/evaluation.md).

The report fixes the users, their hidden games and every strategy's top K. For a sample of users this module rebuilds
the train profile exactly as the evaluation did (`split` from the report + evaluation.train_ratings), pools the top `k`
of the chosen strategies (TREC pooling: a game recommended by several strategies is judged once), asks the judge
(ml.judge) for a 0–3 relevance of every pooled game, and scores the strategies on the judged lists:

- judged@k: mean score, Precision@k (score ≥ 2), graded nDCG@k (gain 2^s − 1, ideal = the user's best judged pool games),
  NovelRelevant@k (score ≥ 2 and not a hidden game: relevant games leave-k-out can't see), and the leave-k-out hits for comparison;
- beyond accuracy on the same lists: novelty (mean self-information −log₂ p of the Steam recommendations share),
  intra-list diversity (1 − mean pairwise cosine of the game vectors) and coverage.

Validation of the judge itself (`--calibrate`, on by default): the hidden liked games, random catalogue games and disliked
train games not shown in the profile are judged too → mean score per group and AUC(hidden liked vs random); a share of the
pairs is judged twice (quadratic weighted κ); scores per popularity bucket show a familiarity bias.

A fusion sweep report (`ml evaluate --fusion`) works too: strategies are then labelled `<fusion>/<strategy>` (popular and random once).
"""

import asyncio
import json
import logging
import math
import random
import statistics
from pathlib import Path

import numpy as np

from ml import recommend as rec
from ml.config import Settings
from ml.db import GameCard, GameRepository, SteamUserGame
from ml.evaluation import POPULAR_POOL, EvaluationParams, train_ratings
from ml.judge import (
    PROFILE_DISLIKED,
    PROFILE_LIKED,
    PROMPT_VERSION,
    Judge,
    JudgeCache,
    JudgeError,
    Judgement,
    pair_key,
    render_candidate,
    render_profile,
)
from ml.vectors import GameVectorStore

log = logging.getLogger(__name__)

DEFAULT_STRATEGIES = ["gresym", "knn", "knn-hybrid-0.05", "knn-hybrid-0.2", "popular", "random"]
SINGLE = ("popular", "random")  # identical across fusion specs
RELEVANT = 2
PAIRED_STRATEGY = "knn-hybrid-0.05"  # the reference of the paired comparison: this strategy of the report's reference fusion
HEAD = 1000  # popularity buckets of the bias check: head = top 1000, mid = rest of the top POPULAR_POOL, tail = others
BOOTSTRAP = 2000
CARD_BATCH = 2000


def _bootstrap_ci(values: list[float], rng: np.random.Generator) -> list[float]:
    if len(values) < 2:
        return [float("nan"), float("nan")]
    array = np.asarray(values, dtype=float)
    means = array[rng.integers(0, len(array), size=(BOOTSTRAP, len(array)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _auc(positive: list[int], negative: list[int]) -> float | None:
    """P(score of a positive > score of a negative), ties half (Mann–Whitney)."""
    if not positive or not negative:
        return None
    pos, neg = np.asarray(positive)[:, None], np.asarray(negative)[None, :]
    return float(((pos > neg).sum() + 0.5 * (pos == neg).sum()) / (pos.size * neg.size))


def _weighted_kappa(a: list[int], b: list[int], categories: int = 4) -> float | None:
    """Quadratic weighted Cohen's κ between two ratings of the same pairs."""
    if len(a) < 2:
        return None
    observed = np.zeros((categories, categories))
    for x, y in zip(a, b):
        observed[x, y] += 1
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    i, j = np.indices((categories, categories))
    weights = (i - j) ** 2 / (categories - 1) ** 2
    denominator = (weights * expected).sum()
    return float(1 - (weights * observed).sum() / denominator) if denominator else None


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def _labels(report: dict, strategies: list[str], fusions: list[str] | None) -> list[tuple[str, str, str | None]]:
    """(label, strategy, fusion spec or None) of every list to judge."""
    if "fusions" not in report:
        return [(s, s, None) for s in strategies]
    specs = fusions or list(report["fusions"])
    unknown = [spec for spec in specs if spec not in report["fusions"]]
    if unknown:
        raise JudgeError(f"Fusion specs not in the report: {unknown}")
    labels = [(f"{spec}/{s}", s, spec) for spec in specs for s in strategies if s not in SINGLE]
    return labels + [(s, s, report["reference"]) for s in strategies if s in SINGLE]


def _train(games: list[SteamUserGame], params: EvaluationParams, hidden: set[int]) -> list[SteamUserGame]:
    usable = [g for g in games if g.recommended is not None or g.playtime_minutes >= params.min_playtime_minutes]
    return [g for g in usable if g.game_id not in hidden]


async def run_judge(
    settings: Settings,
    report_path: Path,
    strategies: list[str],
    fusions: list[str] | None,
    users_count: int,
    k: int,
    seed: int,
    repeat_share: float,
    calibrate: bool,
    cache_path: Path,
    dry_run: bool = False,
) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    params = EvaluationParams.model_validate(report["params"])
    labels = _labels(report, strategies, fusions)
    reports = report.get("fusions") or {None: report}
    base = reports[report.get("reference")] if "fusions" in report else report
    if k > params.ks[-1]:
        raise JudgeError(f"k {k} is larger than the report's lists (top {params.ks[-1]})")
    rng = random.Random(f"{seed}:judge")

    games = GameRepository(settings.database)
    store = GameVectorStore.from_settings(settings)
    try:
        users = {user.name: user for user in games.get_steam_users()}
        per_user = {u["name"]: u for u in base["perUser"] if u["name"] in users}
        names = sorted(per_user, key=lambda n: int(n.split("-")[-1]) if n.split("-")[-1].isdigit() else n)
        sample = sorted(rng.sample(names, min(users_count, len(names))), key=names.index)
        per_user_by_spec = {spec: {u["name"]: u for u in r["perUser"]} for spec, r in reports.items()}
        all_ids = games.get_game_ids()
        counts = games.get_recommendation_counts()
        popular_rank = {game_id: i for i, game_id in enumerate(games.get_popular_game_ids(POPULAR_POOL))}

        # Per sampled user: profile games, the lists, the calibration games
        plans = []
        for name in sample:
            user, row = users[name], per_user[name]
            test = set(row["test"])
            train = _train(user.games, params, test | set(row.get("hiddenDislikes", [])))
            playtime = {g.game_id: g.playtime_minutes for g in user.games}
            ratings = train_ratings(train)
            liked = sorted((r for r in ratings if rec.rating_weight(r) > 0),
                           key=lambda r: (-rec.rating_weight(r), -playtime.get(r.game_id, 0), r.game_id))
            disliked = sorted((g for g in train if g.recommended is False), key=lambda g: (-g.playtime_minutes, g.game_id))
            owned = {g.game_id for g in user.games}
            lists: dict[str, list[int]] = {}
            for label, strategy, spec in labels:
                if strategy == "random":
                    random_rng = random.Random(f"{seed}:{name}:judge-random")
                    picked: list[int] = []
                    while len(picked) < k:
                        game_id = random_rng.choice(all_ids)
                        if game_id not in owned and game_id not in picked:
                            picked.append(game_id)
                    lists[label] = picked
                else:
                    lists[label] = per_user_by_spec[spec][name]["recommended"][strategy][:k]
            calibration: dict[str, list[int]] = {}
            if calibrate:
                calibration_rng = random.Random(f"{seed}:{name}:judge-calibration")
                randoms: list[int] = []
                while len(randoms) < len(test):
                    game_id = calibration_rng.choice(all_ids)
                    if game_id not in owned and game_id not in randoms:
                        randoms.append(game_id)
                calibration = {"hiddenLiked": sorted(test), "random": randoms,
                               "dislikedNotShown": [g.game_id for g in disliked[PROFILE_DISLIKED:PROFILE_DISLIKED + 3]]}
            plans.append({
                "name": name, "test": test, "owned": owned,
                "liked": [r.game_id for r in liked], "disliked": [g.game_id for g in disliked],
                "lists": lists, "calibration": calibration,
            })

        needed = sorted({game_id for plan in plans for game_id in
                         [*plan["liked"][:PROFILE_LIKED], *plan["disliked"][:PROFILE_DISLIKED], *(i for l in plan["lists"].values() for i in l),
                          *(i for l in plan["calibration"].values() for i in l)]})
        cards: dict[int, GameCard] = {}
        for start in range(0, len(needed), CARD_BATCH):
            cards.update(games.get_game_cards(needed[start:start + CARD_BATCH]))

        model = settings.judge.model or "unset"
        pairs: list[tuple[str, str, str]] = []
        repeats: list[tuple[str, str]] = []  # (first key, repeat key)
        for plan in plans:
            profile = render_profile([cards[i] for i in plan["liked"] if i in cards],
                                     [cards[i] for i in plan["disliked"] if i in cards])
            plan["profile"] = profile
            plan["keys"] = {}
            pooled = sorted({i for l in plan["lists"].values() for i in l})
            for game_id in sorted(set(pooled) | {i for l in plan["calibration"].values() for i in l}):
                if game_id not in cards:
                    continue
                key = pair_key(model, profile, game_id)
                plan["keys"][game_id] = key
                pairs.append((key, profile, render_candidate(cards[game_id])))
            for game_id in pooled:
                if game_id in cards and rng.random() < repeat_share:
                    repeat_key = pair_key(model, profile, game_id, repeat=1)
                    repeats.append((plan["keys"][game_id], repeat_key))
                    pairs.append((repeat_key, profile, render_candidate(cards[game_id])))

        pooled_total = sum(len({i for l in p["lists"].values() for i in l}) for p in plans)
        log.info("%d users, %d labels, %d pooled games, %d pairs incl. calibration and %d repeats",
                 len(plans), len(labels), pooled_total, len(pairs), len(repeats))
        if dry_run:
            first = plans[0]
            game_id = next(iter(first["keys"]))
            return {"dryRun": True, "users": len(plans), "pairs": len(pairs), "pooled": pooled_total,
                    "repeats": len(repeats), "examplePrompt": f"{first['profile']}\n\n{render_candidate(cards[game_id])}"}

        cache = JudgeCache(cache_path)
        try:
            judge = Judge(settings.judge, cache)
            judgements = await judge.judge_many(pairs)
            asked, failed = judge.asked, judge.failed
        finally:
            cache.close()

        def score(plan: dict, game_id: int) -> int | None:
            key = plan["keys"].get(game_id)
            judgement: Judgement | None = judgements.get(key) if key else None
            return None if judgement is None else judgement.score

        vectors = store.get_vectors(sorted({i for p in plans for l in p["lists"].values() for i in l}))
        total = sum(counts.values()) + len(all_ids)

        metric_rng = np.random.default_rng(seed)
        strategies_out: dict[str, dict] = {}
        per_user: dict[str, dict[str, dict[str, float]]] = {}  # label → user → metric (for the paired comparison)
        for label, strategy, spec in labels:
            per_user[label] = {}
            columns: dict[str, list[float]] = {m: [] for m in
                                               ("meanScore", "precision", "ndcg", "novelRelevant", "hits", "novelty", "ild")}
            missing = 0
            recommended: set[int] = set()
            for plan in plans:
                ids = plan["lists"][label]
                recommended.update(ids)
                scores = [score(plan, i) for i in ids]
                missing += sum(s is None for s in scores)
                judged = [(i, s) for i, s in zip(ids, scores) if s is not None]
                if not judged:
                    continue
                values = [s for _, s in judged]
                columns["meanScore"].append(statistics.fmean(values))
                columns["precision"].append(sum(s >= RELEVANT for s in values) / k)
                columns["novelRelevant"].append(sum(s >= RELEVANT and i not in plan["test"] for i, s in judged) / k)
                columns["hits"].append(sum(i in plan["test"] for i in ids))
                dcg = sum((2 ** s - 1) / math.log2(rank + 2) for rank, s in enumerate(scores) if s is not None)
                pool_scores = sorted((s for i in {i for l in plan["lists"].values() for i in l}
                                      if (s := score(plan, i)) is not None), reverse=True)[:k]
                ideal = sum((2 ** s - 1) / math.log2(rank + 2) for rank, s in enumerate(pool_scores))
                columns["ndcg"].append(dcg / ideal if ideal else 0.0)
                per_user[label][plan["name"]] = {"meanScore": columns["meanScore"][-1], "ndcg": columns["ndcg"][-1],
                                                 "novelRelevant": columns["novelRelevant"][-1]}
                columns["novelty"].append(statistics.fmean(-math.log2((counts.get(i, 0) + 1) / total) for i in ids))
                matrix = np.asarray([vectors[i] for i in ids if i in vectors], dtype=np.float32)
                if len(matrix) > 1:
                    sims = matrix @ matrix.T
                    n = len(matrix)
                    columns["ild"].append(float(1 - (sims.sum() - np.trace(sims)) / (n * (n - 1))))
            summary: dict = {"users": len(columns["meanScore"]), "missingJudgements": missing,
                             "coverage": len(recommended), "fusion": spec, "strategy": strategy}
            for metric, values in columns.items():
                summary[metric] = {"mean": statistics.fmean(values) if values else None,
                                   "ci": _bootstrap_ci(values, metric_rng)}
            source = reports[spec]["strategies"].get(strategy) if spec in reports else None
            summary["leaveKOutHitRate"] = source[str(params.ks[-1])]["hitRate"] if source else None
            strategies_out[label] = summary

        # Paired per user vs the reference label: same users, same pool, so the per-user noise cancels
        reference = next((label for label, strategy, spec in labels
                          if strategy == PAIRED_STRATEGY and spec == report.get("reference")), labels[0][0])
        paired: dict[str, dict] = {}
        for label, _, _ in labels:
            if label == reference:
                continue
            names_both = [n for n in per_user[label] if n in per_user[reference]]
            paired[label] = {"users": len(names_both)}
            for metric in ("meanScore", "ndcg", "novelRelevant"):
                deltas = [per_user[label][n][metric] - per_user[reference][n][metric] for n in names_both]
                paired[label][metric] = {"delta": statistics.fmean(deltas) if deltas else None,
                                         "ci": _bootstrap_ci(deltas, metric_rng)}

        result: dict = {
            "reference": reference, "paired": paired,
            "report": str(report_path), "model": settings.judge.model, "promptVersion": PROMPT_VERSION,
            "k": k, "users": len(plans), "pairs": len(pairs), "asked": asked, "failed": failed,
            "leaveKOutK": params.ks[-1], "strategies": strategies_out,
        }
        with_both = [(s["ndcg"]["mean"], s["leaveKOutHitRate"]) for s in strategies_out.values()
                     if s["ndcg"]["mean"] is not None and s["leaveKOutHitRate"] is not None]
        result["spearmanJudgedNdcgVsLeaveKOut"] = _spearman([a for a, _ in with_both], [b for _, b in with_both])

        if calibrate:
            groups: dict[str, list[int]] = {}
            for plan in plans:
                for group, ids in plan["calibration"].items():
                    groups.setdefault(group, []).extend(s for i in ids if (s := score(plan, i)) is not None)
            buckets: dict[str, list[int]] = {"head": [], "mid": [], "tail": []}
            for plan in plans:
                for game_id in plan["keys"]:
                    if (s := score(plan, game_id)) is None:
                        continue
                    rank = popular_rank.get(game_id)
                    buckets["head" if rank is not None and rank < HEAD else "mid" if rank is not None else "tail"].append(s)
            result["calibration"] = {
                "groups": {g: {"n": len(v), "mean": statistics.fmean(v) if v else None,
                               "distribution": {str(s): v.count(s) for s in range(4)}} for g, v in groups.items()},
                "aucHiddenLikedVsRandom": _auc(groups.get("hiddenLiked", []), groups.get("random", [])),
                "aucHiddenLikedVsDisliked": _auc(groups.get("hiddenLiked", []), groups.get("dislikedNotShown", [])),
                "popularityBuckets": {b: {"n": len(v), "mean": statistics.fmean(v) if v else None}
                                      for b, v in buckets.items()},
            }
        first_scores, repeat_scores = [], []
        for first_key, repeat_key in repeats:
            if first_key in judgements and repeat_key in judgements:
                first_scores.append(judgements[first_key].score)
                repeat_scores.append(judgements[repeat_key].score)
        result["stability"] = {
            "pairs": len(first_scores),
            "exactAgreement": (sum(a == b for a, b in zip(first_scores, repeat_scores)) / len(first_scores)
                               if first_scores else None),
            "weightedKappa": _weighted_kappa(first_scores, repeat_scores),
        }
        return result
    finally:
        games.close()
        store.close()


def format_judged(result: dict) -> str:
    if result.get("dryRun"):
        return (f"Dry run: {result['users']} users, {result['pooled']} pooled games, {result['pairs']} pairs "
                f"({result['repeats']} repeats).\n\nExample prompt (user message):\n\n{result['examplePrompt']}")
    k = result["k"]
    lines = [f"Judge {result['model']} (prompt v{result['promptVersion']}), {result['users']} users, k = {k}; "
             f"{result['pairs']} pairs, {result['asked']} asked now, {result['failed']} failed",
             "",
             f"{'strategy':<32} {'score':>13} {'P@' + str(k):>13} {'nDCG@' + str(k):>13} {'NovelRel@' + str(k):>13} "
             f"{'hits':>5} {'novelty':>7} {'ILD':>5} {'cov':>5} {'LkO HR@' + str(result['leaveKOutK']):>10}"]

    def cell(m: dict) -> str:
        if m["mean"] is None:
            return f"{'–':>13}"
        return f"{m['mean']:.3f}±{(m['ci'][1] - m['ci'][0]) / 2:.3f}" if not math.isnan(m["ci"][0]) else f"{m['mean']:.3f}"

    for label, s in result["strategies"].items():
        hr = s["leaveKOutHitRate"]
        lines.append(f"{label:<32} {cell(s['meanScore']):>13} {cell(s['precision']):>13} {cell(s['ndcg']):>13} "
                     f"{cell(s['novelRelevant']):>13} {s['hits']['mean'] or 0:>5.2f} {s['novelty']['mean'] or 0:>7.2f} "
                     f"{s['ild']['mean'] or 0:>5.2f} {s['coverage']:>5} {f'{hr:.3f}' if hr is not None else '–':>10}")
    if (rho := result.get("spearmanJudgedNdcgVsLeaveKOut")) is not None:
        lines.append(f"\nSpearman(judged nDCG, leave-k-out HR) over strategies: {rho:+.2f}")
    if paired := result.get("paired"):
        lines.append(f"\nPaired vs {result['reference']} (Δ per user [95 % CI]):")
        lines.append(f"{'strategy':<32} {'Δscore':>24} {'ΔnDCG@' + str(k):>24} {'ΔNovelRel@' + str(k):>24}")
        for label, p in paired.items():
            cells = [f"{p[m]['delta']:+.3f} [{p[m]['ci'][0]:+.3f}, {p[m]['ci'][1]:+.3f}]" if p[m]["delta"] is not None
                     else "–" for m in ("meanScore", "ndcg", "novelRelevant")]
            lines.append(f"{label:<32} {cells[0]:>24} {cells[1]:>24} {cells[2]:>24}")
    if calibration := result.get("calibration"):
        lines.append("\nJudge calibration (mean score, distribution 0/1/2/3):")
        for group, g in calibration["groups"].items():
            dist = "/".join(str(g["distribution"][str(s)]) for s in range(4))
            lines.append(f"  {group:<18} n {g['n']:>5}  mean {g['mean'] if g['mean'] is not None else float('nan'):.2f}  [{dist}]")
        for name in ("aucHiddenLikedVsRandom", "aucHiddenLikedVsDisliked"):
            if (value := calibration.get(name)) is not None:
                lines.append(f"  {name}: {value:.3f}")
        lines.append("  by popularity: " + ", ".join(
            f"{b} {v['mean']:.2f} (n {v['n']})" for b, v in calibration["popularityBuckets"].items() if v["mean"] is not None))
    st = result.get("stability") or {}
    if st.get("pairs"):
        lines.append(f"\nStability on {st['pairs']} repeated pairs: exact agreement {st['exactAgreement']:.2%}, "
                     f"quadratic weighted κ {st['weightedKappa']:.3f}")
    return "\n".join(lines)


def judge_command(settings: Settings, args) -> None:
    result = asyncio.run(run_judge(
        settings, args.report, args.strategies, args.fusions, args.users, args.k, args.seed, args.repeat_share,
        not args.no_calibrate, args.cache, args.dry_run,
    ))
    print(format_judged(result))
    if args.json and not result.get("dryRun"):
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
