"""Offline NSFW regressions. Run: .venv/bin/python -m unittest discover -s tests -v."""
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from ml import personal, recommend
from ml.catalogue import CatalogueStats
from ml.config import get_settings
from ml.search import GameSearch
from ml.text_index import GameTextStore, TITLE_FIELD, TEXT_FIELD
from ml.visibility import excluded_game_ids
from ml.worker import RecommendTask


class VisibilityTests(unittest.TestCase):
    def test_default_and_explicit_opt_out_contract(self):
        self.assertTrue(personal.RecommendationsRequest(ratings=[]).hide_nsfw)
        self.assertFalse(personal.RecommendationsRequest(ratings=[], hideNsfw=False).hide_nsfw)
        self.assertTrue(RecommendTask(recommendationId=1, ratings=[]).hide_nsfw)

    def test_live_classification_and_connection_cleanup(self):
        with patch('ml.visibility.GameRepository') as factory:
            repo = factory.return_value
            repo.get_nsfw_game_ids.side_effect = [[1], [2]]
            self.assertEqual(excluded_game_ids(None, True), [1])
            self.assertEqual(excluded_game_ids(None, True), [2])
            self.assertEqual(repo.close.call_count, 2)
            repo.get_nsfw_game_ids.side_effect = RuntimeError('database unavailable')
            with self.assertRaises(RuntimeError):
                excluded_game_ids(None, True)
            self.assertEqual(repo.close.call_count, 3)
        with patch('ml.visibility.GameRepository') as factory:
            self.assertEqual(excluded_game_ids(None, False), [])
            factory.assert_not_called()

    def test_both_bm25_fields_filter_before_limit(self):
        texts = GameTextStore.__new__(GameTextStore)
        texts._collection = 'test'
        texts._client = MagicMock()
        texts._client.search.return_value = [[{'id': 7}]]
        self.assertEqual(texts.search('game', 20, [1, 2]), {TITLE_FIELD: [7], TEXT_FIELD: [7]})
        for call in texts._client.search.call_args_list:
            self.assertEqual(call.kwargs['filter'], 'id not in [1, 2]')
            self.assertEqual(call.kwargs['limit'], 20)

    def test_search_passes_exclusions_to_all_retrievers(self):
        settings = get_settings().model_copy(deep=True)
        settings.search.dense = True
        texts, vectors = MagicMock(), MagicMock()
        texts.search.return_value = {TITLE_FIELD: [3], TEXT_FIELD: [3, 4]}
        vectors.search.return_value = [(4, 0.9)]
        search = GameSearch(settings, texts, vectors)
        search._embedder = MagicMock()
        search._embedder.embed_query.return_value = [1.0] * 768
        search.search('game', False, 1, [1, 2])
        self.assertEqual(texts.search.call_args.kwargs['exclude_ids'], [1, 2])
        self.assertEqual(vectors.search.call_args.kwargs['exclude_ids'], [1, 2])

    def test_similar_hides_source_and_filters_candidates(self):
        store = MagicMock()
        self.assertIsNone(recommend.similar(store, 1, 20, [1, 2]))
        store.get_vectors.assert_not_called()
        store.get_vectors.return_value = {3: [1.0, 0.0]}
        recommend.similar(store, 3, 20, [1, 2])
        self.assertEqual(store.search.call_args.kwargs['exclude_ids'], [1, 2, 3])

    def test_hidden_taste_signals_kept_popularity_and_knn_filtered(self):
        request = personal.RecommendationsRequest(
            ratings=[{'gameId': 1, 'recommend': True}], planned=[2], rejected=[3], limit=2)
        with patch('ml.personal.rec.recommend') as compute:
            compute.return_value = SimpleNamespace(items=[], used_ratings=1)
            personal.recommend(MagicMock(), MagicMock(), request, [1, 2, 3, 4])
            self.assertEqual([r.game_id for r in compute.call_args.args[2]], [1, 2, 3])
            self.assertEqual(compute.call_args.kwargs['exclude_ids'], [1, 2, 3, 4])
        # The excluded popular game must not consume the popularity candidate budget.
        stats = CatalogueStats(np.zeros(2), {4: 100, 5: 90}, 100,
                               [4, 5], np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32))
        store = MagicMock()
        store.get_vectors.side_effect = [{1: [1.0, 0.0]}, {}]
        store.search_many.return_value = [[]]
        result = recommend.recommend(store, stats, [recommend.Rating(1, recommend=True)], 2,
                                     mainstream=100, exclude_ids=[4])
        self.assertEqual([i for i, _ in result.items], [5])
        self.assertIn(4, store.search_many.call_args.args[2])

    def test_ten_thousand_exclusion_expression(self):
        texts = GameTextStore.__new__(GameTextStore)
        texts._collection = 'test'
        texts._client = MagicMock()
        texts._client.search.return_value = [[]]
        started = time.perf_counter()
        texts.search('game', 200, list(range(10000)))
        expression = texts._client.search.call_args.kwargs['filter']
        self.assertIn('9999', expression)
        self.assertLess(len(expression), 100_000)
        print(f'10k exclusion expression: {len(expression)} bytes, {(time.perf_counter() - started)*1000:.1f} ms locally')


if __name__ == '__main__':
    unittest.main()
