"""Which reviews of a game get embedded, and how much they weigh in the game vector.

The game vector describes what the game is like, so we want reviews that talk about the game:
long enough, rated helpful by Steam users, and mixed positive/negative in the game's real proportion
(negative reviews often describe a game most concretely). Selection is deterministic: the staleness
check re-runs it and compares the result with the review ids stored next to the vector.
"""

import re
from dataclasses import dataclass

from ml.config import ReviewSettings
from ml.db import ReviewRow

# Steam BBCode: [b], [/b], [h1], [url=...], [spoiler], [list], [*], ...
_BBCODE = re.compile(r"\[/?[a-z0-9*]+(?:=[^\]]*)?\]", re.IGNORECASE)
_SPACES = re.compile(r"[ \t\r\f\v\xa0]+")
_NEWLINES = re.compile(r"\n{2,}")
# Steam masks profanity with hearts
_MASKED = re.compile(r"♥+")
# The same character 5+ times ("!!!!!!", "ахахахаха" isn't caught, ASCII-art lines are)
_REPEATS = re.compile(r"(.)\1{4,}")


def clean_review(text: str | None) -> str:
    if not text:
        return ""
    text = _MASKED.sub("", _BBCODE.sub("", text))
    text = _REPEATS.sub(r"\1\1\1", text)
    lines = (_SPACES.sub(" ", line).strip() for line in text.split("\n"))
    return _NEWLINES.sub("\n", "\n".join(line for line in lines if line)).strip()


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    space = cut.rfind(" ")
    return (cut[:space] if space > max_chars * 0.7 else cut).rstrip(" ,.;:-") + "…"


@dataclass(frozen=True)
class SelectedReview:
    id: int
    recommended: bool
    text: str  # cleaned and truncated: exactly what gets embedded


def positive_share(steam_positive: int | None, steam_negative: int | None) -> float | None:
    """Share of positive reviews on Steam (all languages), None if unknown."""
    if steam_positive is None or steam_negative is None or steam_positive + steam_negative == 0:
        return None
    return steam_positive / (steam_positive + steam_negative)


def _helpfulness(review: ReviewRow, text: str) -> tuple:
    # Without Steam metadata (reviews parsed before it was stored) longer reviews win, which is a fair proxy
    return (
        review.weighted_vote_score or 0.0,
        review.votes_up or 0,
        len(text),
        -review.id,
    )


def select_reviews(reviews: list[ReviewRow], share: float | None, settings: ReviewSettings) -> list[SelectedReview]:
    """Top reviews by helpfulness, split into positive/negative by the game's share of positive reviews.

    `share` is Steam's real ratio. Our DB only keeps a sample of each kind, so when it's unknown the
    split is 50/50. At least one negative review is taken when there are any.
    """
    seen: set[str] = set()
    positive: list[tuple[tuple, SelectedReview]] = []
    negative: list[tuple[tuple, SelectedReview]] = []
    for review in reviews:
        text = clean_review(review.content)
        if len(text) < settings.min_chars:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        selected = SelectedReview(review.id, bool(review.recommended), _truncate(text, settings.max_chars))
        (positive if review.recommended else negative).append((_helpfulness(review, text), selected))

    positive.sort(key=lambda item: item[0], reverse=True)
    negative.sort(key=lambda item: item[0], reverse=True)

    limit = settings.max_selected
    want_negative = round(limit * (1 - share)) if share is not None else limit // 2
    if negative:
        want_negative = max(1, want_negative)
    take_negative = min(want_negative, len(negative))
    take_positive = min(limit - take_negative, len(positive))
    # Not enough of one kind: fill up with the other
    take_negative = min(limit - take_positive, len(negative))

    chosen = positive[:take_positive] + negative[:take_negative]
    return sorted((item[1] for item in chosen), key=lambda r: r.id)


def reviews_weight(selected_count: int, settings: ReviewSettings) -> float:
    """α in normalize(v_desc + α·v_reviews): grows with the number of reviews up to weight_max."""
    if selected_count == 0:
        return 0.0
    return settings.weight_max * min(1.0, selected_count / settings.max_selected)
