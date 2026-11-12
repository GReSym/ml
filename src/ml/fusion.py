"""How a game's description vector d and review vector r become its search vector (docs/ml.md, "Fusion").

Both are unit vectors from the embedding model; r is the normalized mean of the selected reviews (0 without reviews).
`ramp = min(1, selected reviews / reviews.max_selected)`, α = `reviews.weight_max`:

- `sum`:            normalize(d + α·ramp·r)                          (until 2026-11-12)
- `rcenter`:        the same with r replaced by normalize(r − r̄)
- `concat`:         normalize([d, √(α·ramp)·r])                      (twice the dimension)
- `rcenter-concat`: normalize([d, √(α·ramp)·normalize(r − r̄)])       (default since 2026-11-12)

r̄ is the mean review embedding of the catalogue. Review texts share a direction ("a Steam review", mean cosine 0.50 between
different games' reviews vs 0.32 for descriptions); centering removes it. With `concat`, cosine(a, b) = (d·d' + α·ramp·r·r') /
norms: description and review similarities are added without the cross terms d·r'. Chosen by the fusion sweep
(docs/evaluation.md). r̄ is frozen in a file (`reviews.mean_path`, written by `ml recombine`), so new games are centered
exactly like the rest of the collection.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import numpy as np

log = logging.getLogger(__name__)

FusionKind = Literal["sum", "rcenter", "concat", "rcenter-concat"]
FUSION_KINDS: tuple[str, ...] = get_args(FusionKind)


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms > 0, norms, 1)


@dataclass(frozen=True)
class FusionSpec:
    kind: str
    alpha: float

    @property
    def name(self) -> str:
        return f"{self.kind}:{self.alpha:g}"

    @property
    def centered(self) -> bool:
        return self.kind.startswith("rcenter")

    @property
    def concatenated(self) -> bool:
        return self.kind.endswith("concat")

    def search_dimension(self, model_dimension: int) -> int:
        return 2 * model_dimension if self.concatenated else model_dimension

    @classmethod
    def parse(cls, value: str) -> FusionSpec:
        kind, sep, alpha = value.strip().partition(":")
        if kind not in FUSION_KINDS or not sep:
            raise ValueError(f"fusion spec {value!r}: expected <{'|'.join(FUSION_KINDS)}>:<alpha>")
        a = float(alpha)
        if a < 0 or a > 10:
            raise ValueError(f"fusion spec {value!r}: alpha must be in [0, 10]")
        return cls(kind, a)

    def weights(self, review_counts: np.ndarray, max_selected: int) -> np.ndarray:
        """α·ramp per game (stored as `reviews_weight`)."""
        return (self.alpha * np.minimum(1.0, np.asarray(review_counts, dtype=np.float32) / max_selected)).astype(np.float32)

    def fuse(self, desc: np.ndarray, reviews: np.ndarray, review_counts: np.ndarray, max_selected: int,
             review_mean: np.ndarray | None) -> np.ndarray:
        """Search vectors (unit rows) of n games: desc and reviews are n×dim, reviews rows are 0 without reviews."""
        has_reviews = np.asarray(review_counts) > 0
        weight = self.weights(review_counts, max_selected)[:, None]
        if self.centered:
            if review_mean is None:
                raise ValueError(f"Fusion {self.kind} needs the mean review vector (reviews.mean_path)")
            reviews = unit_rows(reviews - review_mean) * has_reviews[:, None]
        if self.concatenated:
            return unit_rows(np.concatenate([desc, np.sqrt(weight) * reviews], axis=1)).astype(np.float32)
        return unit_rows(desc + weight * reviews).astype(np.float32)

    def fuse_one(self, desc: np.ndarray, reviews: np.ndarray | None, review_count: int, max_selected: int,
                 review_mean: np.ndarray | None) -> np.ndarray:
        desc = np.asarray(desc, dtype=np.float32)
        reviews = np.zeros_like(desc) if reviews is None else np.asarray(reviews, dtype=np.float32)
        count = review_count if np.any(reviews) else 0
        return self.fuse(desc[None, :], reviews[None, :], np.asarray([count]), max_selected, review_mean)[0]

    def query(self, vector: list[float]) -> list[float]:
        """A text query embedding (model dimension) as a search vector: with `concat` it only meets the description half."""
        return list(vector) + [0.0] * len(vector) if self.concatenated else list(vector)


def load_review_mean(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Mean review vector {path} is missing: `ml recombine` writes it")
    return np.load(path).astype(np.float32)


def save_review_mean(path: Path, mean: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, mean.astype(np.float32))
    log.info("Mean review vector written to %s", path)
