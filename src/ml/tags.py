"""Tag-only recommendations, the evaluation's answer to "why not just use the player's favourite tags?".

Steam tags aren't part of the embedded text (ml/text.py), so this is an independent content baseline. Every game is a
binary tag vector weighted by IDF (store features like "Семейный доступ" are on almost every game and get ~0 weight).
The whole catalogue is scored per user with bincount over the (game, tag) pairs, no candidate pool.
"""

import numpy as np

# tags-top: the user's most characteristic tags a game is matched against
TOP_TAGS = 10


class TagIndex:
    def __init__(self, pair_game_ids: list[int], pair_tag_ids: list[int], game_ids: list[int]):
        """`game_ids` = the catalogue (scores are returned in `self.ids` order); pairs of other games are dropped."""
        self.ids = np.unique(np.asarray(game_ids, dtype=np.int64))
        pair_games = np.asarray(pair_game_ids, dtype=np.int64)
        rows = np.searchsorted(self.ids, pair_games).clip(max=len(self.ids) - 1)
        known = self.ids[rows] == pair_games
        tags, cols = np.unique(np.asarray(pair_tag_ids, dtype=np.int64)[known], return_inverse=True)
        order = np.argsort(rows[known], kind="stable")
        self._rows, self._cols = rows[known][order], cols[order]
        self.tag_ids = tags
        self.idf = np.log(len(self.ids) / np.bincount(self._cols, minlength=len(tags)))
        # Pair weight = idf / game norm, so a game's row is its unit TF-IDF vector
        norms = np.sqrt(np.bincount(self._rows, weights=self.idf[self._cols] ** 2, minlength=len(self.ids)))
        self._pair_weight = self.idf[self._cols] / np.where(norms > 0, norms, 1)[self._rows]
        self._starts = np.searchsorted(self._rows, np.arange(len(self.ids) + 1))

    def rows(self, game_ids: list[int] | np.ndarray) -> np.ndarray:
        """Positions in `self.ids` of the given games; games outside the catalogue are left out."""
        game_ids = np.asarray(game_ids, dtype=np.int64)
        if not len(game_ids):
            return np.empty(0, dtype=np.int64)
        rows = np.searchsorted(self.ids, game_ids).clip(max=len(self.ids) - 1)
        return rows[self.ids[rows] == game_ids]

    def _pairs(self, rows: np.ndarray) -> np.ndarray:
        """Pair indices of the given games' tags."""
        if not len(rows):
            return np.empty(0, dtype=np.int64)
        return np.concatenate([np.arange(self._starts[r], self._starts[r + 1]) for r in rows])

    def profile(self, game_ids: list[int], weights: list[float]) -> np.ndarray:
        """Weighted sum of the games' unit TF-IDF vectors (one value per tag)."""
        rows = self.rows(game_ids)
        weight_of = dict(zip(np.asarray(game_ids, dtype=np.int64).tolist(), weights))
        pairs = self._pairs(rows)
        game_weight = np.asarray([weight_of[int(self.ids[r])] for r in self._rows[pairs]], dtype=np.float64)
        return np.bincount(self._cols[pairs], weights=self._pair_weight[pairs] * game_weight, minlength=len(self.tag_ids))

    def scores(self, profile: np.ndarray) -> np.ndarray:
        """Cosine of every catalogue game to the profile (up to the profile's norm, a per-user constant)."""
        return np.bincount(self._rows, weights=self._pair_weight * profile[self._cols], minlength=len(self.ids))

    def top_tags(self, profile: np.ndarray, count: int = TOP_TAGS) -> np.ndarray:
        """Column indices of the profile's strongest tags (its values already include IDF)."""
        return np.argsort(-profile, kind="stable")[:count]

    def match_counts(self, columns: np.ndarray) -> np.ndarray:
        """How many of the given tags every catalogue game has."""
        return np.bincount(self._rows, weights=np.isin(self._cols, columns).astype(np.float64), minlength=len(self.ids))
