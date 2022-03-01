import unittest

import pandas as pd
import torch
from transformers import BatchEncoding

import data.reference_aggregator as ra


class TestReferenceAggregator(unittest.TestCase):
    def test_gen_sorted_distinct_list(self):
        series = pd.Series([1, 0, 0])
        self.assertEqual([0, 1], ra.gen_sorted_distinct_list(series))

    def test_collect_references_and_word_vectors(self):
        df = pd.DataFrame({'tokens': ['Hello', 'world', 'Hello', 'Hello'],
                           'reference-id': [0, 0, 1, 1],
                           'word-vector-id': [0, 1, 2, 3]})
        df_expected = pd.DataFrame({'tokens': ['Hello', 'world'],
                                    'reference-id': [[0, 1], [0]],
                                    'word-vector-id': [[0, 2, 3], [1]]})
        df_result = ra.collect_references_and_word_vectors(df)
        pd.testing.assert_frame_equal(df_result, df_expected)

    def test_gen_ids_for_tokens_and_references(self):
        encodings = [BatchEncoding({'input_ids': torch.tensor([[0, 3]])}),
                     BatchEncoding({'input_ids': torch.tensor([[7]])})]
        df_expected = pd.DataFrame({'token': [0, 3, 7],
                                    'reference-id': [0, 0, 1],
                                    'word-vector-id': [0, 1, 2]})

        df_result = ra.gen_ids_for_tokens_and_references(encodings)
        pd.testing.assert_frame_equal(df_result, df_expected)

    def test_concat_word_vectors(self):
        word_vectors = [torch.tensor([[1, 1], [1, 1]]), torch.tensor([[0, 0]])]
        expected = torch.tensor([[1, 1], [1, 1], [0, 0]])
        result = ra.concat_word_vectors(word_vectors)
        self.assertTrue(torch.equal(expected, result))


if __name__ == '__main__':
    unittest.main()
