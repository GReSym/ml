"""Finds games whose vector is out of date, for the admin's "pending re-vectorization" list.

Compares the snapshot stored next to each vector (ml.vectors) with the current state of MariaDB.
Nothing is embedded here and nothing is queued: the admin picks games and requests vectorization.

Cheap checks run for every game; texts and reviews are only read for games whose rows were
updated after their vector was built (the parser bumps updated_at on every re-parse, so a newer
timestamp only means "maybe changed", and the hash / review selection decides).
"""

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum

from ml.config import Settings
from ml.db import GameChangeInfo, GameRepository
from ml.reviews import positive_share
from ml.text import TEXT_VERSION
from ml.vectorize import build_input
from ml.vectors import GameVectorStore, VectorSnapshot

_CHUNK = 500


class Reason(StrEnum):
    DESCRIPTION = "description"  # title/description text changed
    SENTIMENT = "sentiment"  # Steam positive share moved
    REVIEWS = "reviews"  # the reviews we'd embed now differ from the embedded ones
    GROWTH = "growth"  # many new reviews on Steam
    OUTDATED = "outdated"  # built with an older TEXT_VERSION
    MISSING = "missing"  # no vector yet
    DELETED = "deleted"  # vector of a game that's no longer in MariaDB


# Base priority of each reason; a game's score is the sum over its reasons (plus a per-reason magnitude)
_PRIORITY = {
    Reason.DESCRIPTION: 3.0,
    Reason.SENTIMENT: 2.0,
    Reason.REVIEWS: 1.0,
    Reason.GROWTH: 1.0,
    Reason.OUTDATED: 0.5,
    Reason.MISSING: 0.2,
    Reason.DELETED: 0.1,
}


@dataclass
class StaleGame:
    game_id: int
    vectorized_at: int | None
    reasons: list[Reason] = field(default_factory=list)
    score: float = 0.0
    details: dict[str, float | int | None] = field(default_factory=dict)

    def add(self, reason: Reason, magnitude: float = 0.0) -> None:
        self.reasons.append(reason)
        self.score += _PRIORITY[reason] + magnitude


@dataclass
class StaleReport:
    collection: str
    text_version: int
    vectors: int
    games: int
    items: list[StaleGame]  # best first

    def counts(self) -> dict[Reason, int]:
        result = dict.fromkeys(Reason, 0)
        for item in self.items:
            for reason in item.reasons:
                result[reason] += 1
        return result


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _chunks(ids: list[int]) -> Iterable[list[int]]:
    for start in range(0, len(ids), _CHUNK):
        yield ids[start : start + _CHUNK]


def find_stale(settings: Settings, store: GameVectorStore, games: GameRepository) -> StaleReport:
    rules = settings.staleness
    infos: dict[int, GameChangeInfo] = {info.id: info for info in games.get_change_info()}
    snapshots: dict[int, VectorSnapshot] = {snap.id: snap for snap in store.snapshots()}

    stale: dict[int, StaleGame] = {}

    def entry(game_id: int) -> StaleGame:
        if game_id not in stale:
            snap = snapshots.get(game_id)
            stale[game_id] = StaleGame(game_id, None if snap is None else snap.updated_at)
        return stale[game_id]

    for game_id in snapshots.keys() - infos.keys():
        entry(game_id).add(Reason.DELETED)

    to_inspect: list[int] = []  # rows updated after vectorization: compare texts and review selections
    for game_id, info in infos.items():
        snap = snapshots.get(game_id)
        if snap is None:
            entry(game_id).add(Reason.MISSING)
            continue
        if snap.text_version < TEXT_VERSION:
            # Will be rebuilt from scratch anyway, the finer checks would only add noise
            entry(game_id).add(Reason.OUTDATED)
            continue

        reviews_touched = (info.reviews_updated_at or 0) > snap.updated_at or (
            info.reviews_updated_at is None and snap.review_ids  # all embedded reviews are gone
        )
        if info.updated_at > snap.updated_at or reviews_touched:
            to_inspect.append(game_id)

        total_now = (
            None if info.steam_positive is None or info.steam_negative is None
            else info.steam_positive + info.steam_negative
        )
        share_now = positive_share(info.steam_positive, info.steam_negative)

        if share_now is not None and snap.steam_positive_share is not None and (total_now or 0) >= rules.sentiment_min_total:
            shift = abs(share_now - snap.steam_positive_share)
            if shift >= rules.sentiment_shift:
                item = entry(game_id)
                item.add(Reason.SENTIMENT, shift)
                item.details.update(shareBefore=round(snap.steam_positive_share, 3), shareNow=round(share_now, 3))

        if total_now is not None and snap.steam_reviews_total is not None:
            new = total_now - snap.steam_reviews_total
            if new >= rules.growth_min_new and new >= rules.growth_ratio * snap.steam_reviews_total:
                item = entry(game_id)
                item.add(Reason.GROWTH, min(1.0, new / max(1, snap.steam_reviews_total)))
                item.details.update(steamReviewsBefore=snap.steam_reviews_total, steamReviewsNow=total_now)

    for ids in _chunks(to_inspect):
        reviews = games.get_reviews(ids)
        for game in games.get_games(ids):
            snap = snapshots[game.id]
            current = build_input(game, reviews.get(game.id, []), settings.reviews)

            if current.desc_hash != snap.desc_hash:
                entry(game.id).add(Reason.DESCRIPTION)

            before, now = set(snap.review_ids), {r.id for r in current.reviews}
            overlap = _jaccard(before, now)
            if overlap < rules.min_overlap and max(len(before), len(now)) >= rules.min_selected:
                item = entry(game.id)
                item.add(Reason.REVIEWS, 1.0 - overlap)
                item.details.update(reviewsBefore=len(before), reviewsNow=len(now), overlap=round(overlap, 3))

    items = sorted(stale.values(), key=lambda s: (-s.score, s.game_id))
    return StaleReport(store.collection, TEXT_VERSION, len(snapshots), len(infos), items)


def filter_items(items: list[StaleGame], reasons: Iterable[str] | None) -> list[StaleGame]:
    if not reasons:
        return items
    wanted = {Reason(r) for r in reasons}
    return [item for item in items if wanted.intersection(item.reasons)]


def print_report(settings: Settings, limit: int, reasons: list[str] | None, ids_only: bool) -> None:
    store = GameVectorStore(settings.milvus, settings.model)
    games = GameRepository(settings.database)
    try:
        store.ensure_collection()
        started = time.monotonic()
        report = find_stale(settings, store, games)
        elapsed = time.monotonic() - started
    finally:
        games.close()
        store.close()

    items = filter_items(report.items, reasons)
    if ids_only:
        print(",".join(str(item.game_id) for item in items[:limit]))
        return

    print(f"{report.collection} (text v{report.text_version}): {report.vectors} vectors, "
          f"{report.games} games, checked in {elapsed:.1f} s")
    print("  " + ", ".join(f"{reason}: {count}" for reason, count in report.counts().items()))
    print(f"{len(items)} matching, showing {min(limit, len(items))}:")
    for item in items[:limit]:
        vectorized = time.strftime("%Y-%m-%d %H:%M", time.localtime(item.vectorized_at)) if item.vectorized_at else "-"
        details = " ".join(f"{k}={v}" for k, v in item.details.items())
        print(f"  {item.game_id:>6}  {item.score:5.2f}  {vectorized:16}  {','.join(item.reasons):28} {details}")
