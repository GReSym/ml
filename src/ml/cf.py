"""Collaborative-filtering baselines for the offline evaluation (`ml evaluate --cf`, docs/evaluation.md).

Trained only on the collected Steam users: a binary implicit matrix X (users × games) of every user's liked *train* games
(played ≥ minPlaytimeMinutes or reviewed positively; hidden games are not in it), one model per evaluation run.

- ItemCF: classic item-based CF with cosine similarity on co-occurrence, sim(i, j) = c_ij / √(n_i·n_j), score(i) = Σ_{j∈u} sim(i, j).
  Computed exactly without the item × item matrix: score = (1/√n) ⊙ Xᵀ(X[:, u] · 1/√n_u), two sparse products per user.
- EASE (Steck 2019, "Embarrassingly Shallow Autoencoders"): B = I − P·diag(1/diag P), P = (XᵀX + λI)⁻¹, score = x_u·B.
  A strong linear CF baseline on small data; the dense P limits it to games with ≥ EASE_MIN_USERS users.

- CommunityPopular: the games most played by the collected users, not personalized (a control: how much of CF is just
  "what this social circle plays", since the users come from one friend graph).

A game outside a model's item set scores 0: CF can't recommend games no other collected user played (cold start).
"""

import logging
import time
from collections import Counter

import numpy as np
from scipy import sparse

log = logging.getLogger(__name__)

ITEM_CF_MIN_USERS = 2
EASE_MIN_USERS = 5


class CfData:
    """Binary user × item matrix over the games with at least `min_users` users; `rows` = liked game ids per user."""

    def __init__(self, rows: list[list[int]], min_users: int) -> None:
        counts = Counter(game_id for row in rows for game_id in set(row))
        self.items = np.asarray(sorted(g for g, c in counts.items() if c >= min_users), dtype=np.int64)
        self._column = {int(g): i for i, g in enumerate(self.items)}
        indptr, indices = [0], []
        for row in rows:
            indices.extend(sorted({self._column[g] for g in row if g in self._column}))
            indptr.append(len(indices))
        self.matrix = sparse.csr_matrix(
            (np.ones(len(indices), dtype=np.float32), np.asarray(indices, dtype=np.int64), np.asarray(indptr)),
            shape=(len(rows), len(self.items)),
        )

    def columns(self, game_ids: list[int]) -> np.ndarray:
        return np.asarray([c for g in game_ids if (c := self._column.get(int(g))) is not None], dtype=np.int64)

    def lookup(self, scores: np.ndarray, game_ids: list[int] | np.ndarray) -> np.ndarray:
        """Scores of the given games, 0 for games outside the item set."""
        return np.asarray([scores[c] if (c := self._column.get(int(g))) is not None else 0.0 for g in game_ids],
                          dtype=np.float64)

    def top(self, scores: np.ndarray, excluded: set[int], k: int) -> list[int]:
        """The k best-scoring games not in `excluded`, ties by game id."""
        scores = scores.astype(np.float64, copy=True)
        blocked = self.columns(sorted(excluded))
        scores[blocked] = -np.inf
        k = min(k, int(np.isfinite(scores).sum()))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.lexsort((self.items[top], -scores[top]))]
        return [int(self.items[i]) for i in top]


class ItemCF:
    def __init__(self, rows: list[list[int]], min_users: int = ITEM_CF_MIN_USERS) -> None:
        self.data = CfData(rows, min_users)
        x = self.data.matrix
        self._csc = x.tocsc()
        self._xt = x.T.tocsr()
        counts = np.asarray(x.sum(axis=0)).ravel()
        self._inv_sqrt = np.where(counts > 0, 1 / np.sqrt(np.maximum(counts, 1)), 0).astype(np.float32)
        log.info("Item CF: %d users × %d games (≥ %d users), %d interactions", x.shape[0], x.shape[1], min_users, x.nnz)

    def scores(self, liked: list[int]) -> np.ndarray:
        cols = self.data.columns(liked)
        if cols.size == 0:
            return np.zeros(len(self.data.items), dtype=np.float32)
        user_weights = self._csc[:, cols] @ self._inv_sqrt[cols]  # per user: Σ_{j∈u} X_vj / √n_j
        return (self._xt @ user_weights) * self._inv_sqrt


class Ease:
    def __init__(self, rows: list[list[int]], lam: float, min_users: int = EASE_MIN_USERS) -> None:
        started = time.monotonic()
        self.data = CfData(rows, min_users)
        x = self.data.matrix
        # In place where possible: n² float64 is ~1.4 GB for 13k games, so the peak stays at about two such matrices
        gram = (x.T @ x).toarray().astype(np.float64)
        gram[np.diag_indices_from(gram)] += lam
        p = np.linalg.inv(gram)
        del gram
        p /= -np.diag(p)  # column j divided by −P_jj
        p[np.diag_indices_from(p)] = 0.0
        self._b = p.astype(np.float32)
        del p
        log.info("EASE: %d users × %d games (≥ %d users), λ %g, fitted in %.0f s",
                 x.shape[0], x.shape[1], min_users, lam, time.monotonic() - started)

    def scores(self, liked: list[int]) -> np.ndarray:
        cols = self.data.columns(liked)
        if cols.size == 0:
            return np.zeros(len(self.data.items), dtype=np.float32)
        return self._b[cols].sum(axis=0)


class CommunityPopular:
    """Not personalized: the games most played by the collected users (train rows), the "what this community plays" control
    for the CF models (Steam-wide popularity is the `popular` strategy)."""

    def __init__(self, rows: list[list[int]]) -> None:
        self.data = CfData(rows, 1)
        self._counts = np.asarray(self.data.matrix.sum(axis=0)).ravel().astype(np.float32)

    def scores(self, liked: list[int]) -> np.ndarray:
        return self._counts
