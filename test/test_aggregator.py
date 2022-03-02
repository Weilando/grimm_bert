import unittest

import pandas as pd
import torch
from transformers import BatchEncoding

import data.aggregator as da


class TestAggregator(unittest.TestCase):
    def test_gen_sorted_distinct_list(self):
        series = pd.Series([1, 0, 0])
        self.assertEqual([0, 1], da.gen_sorted_distinct_list(series))

    def test_collect_references_and_word_vectors(self):
        df = pd.DataFrame({'token': [42, 7, 42, 42],
                           'reference-id': [0, 0, 1, 1],
                           'word-vector-id': [0, 1, 2, 3]})
        df_expected = pd.DataFrame({'token': [7, 42],
                                    'reference-id': [[0], [0, 1]],
                                    'word-vector-id': [[1], [0, 2, 3]]})
        df_result = da.collect_references_and_word_vectors(df)
        pd.testing.assert_frame_equal(df_result, df_expected)

    def test_gen_ids_for_tokens_and_references(self):
        encodings = [BatchEncoding({'input_ids': torch.tensor([[0, 3]])}),
                     BatchEncoding({'input_ids': torch.tensor([[7]])})]
        df_expected = pd.DataFrame({'token': [0, 3, 7],
                                    'reference-id': [0, 0, 1],
                                    'word-vector-id': [0, 1, 2]})

        df_result = da.gen_ids_for_tokens_and_references(encodings)
        pd.testing.assert_frame_equal(df_result, df_expected)

    def test_concat_word_vectors(self):
        word_vectors = [torch.tensor([[1, 1], [1, 1]]), torch.tensor([[0, 0]])]
        expected = torch.tensor([[1, 1], [1, 1], [0, 0]])
        result = da.concat_word_vectors(word_vectors)
        self.assertTrue(torch.equal(expected, result))


if __name__ == '__main__':
    unittest.main()
