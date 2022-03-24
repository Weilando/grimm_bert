from unittest import main, TestCase

import pandas as pd
import torch
from transformers import BatchEncoding

from analysis import aggregator as ag


class TestAggregator(TestCase):
    def test_gen_sorted_distinct_list(self):
        series = pd.Series([1, 0, 0])
        self.assertEqual([0, 1], ag.gen_sorted_distinct_list(series))

    def test_agg_references_and_word_vectors(self):
        df = pd.DataFrame({'token': [42, 7, 42, 42],
                           'reference_id': [0, 0, 1, 1],
                           'word_vector_id': [0, 1, 2, 3]})
        df_expected = pd.DataFrame({'token': [7, 42],
                                    'reference_id': [[0], [0, 1]],
                                    'word_vector_id': [[1], [0, 2, 3]]})
        df_result = ag.agg_references_and_word_vectors(df, by='token')
        pd.testing.assert_frame_equal(df_result, df_expected)

    def test_gen_ids_for_vectors_and_references(self):
        encodings = [BatchEncoding({'input_ids': torch.tensor([[0, 3]])}),
                     BatchEncoding({'input_ids': torch.tensor([[7]])})]
        df_expected = pd.DataFrame({'token': [0, 3, 7],
                                    'reference_id': [0, 0, 1],
                                    'word_vector_id': [0, 1, 2]})

        df_result = ag.gen_ids_for_vectors_and_references(encodings)
        pd.testing.assert_frame_equal(df_result, df_expected)

    def test_concat_word_vectors(self):
        word_vectors = [torch.tensor([[1, 1], [1, 1]]), torch.tensor([[0, 0]])]
        expected = torch.tensor([[1, 1], [1, 1], [0, 0]])
        result = ag.concat_word_vectors(word_vectors)
        self.assertTrue(torch.equal(expected, result))


if __name__ == '__main__':
    main()
