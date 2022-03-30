from unittest import main, TestCase

import pandas as pd
import torch

from analysis import aggregator as ag


class TestAggregator(TestCase):
    def test_collect_references_and_word_vectors(self):
        df = pd.DataFrame({'token': [42, 7, 42, 42],
                           'reference_id': [0, 0, 1, 1],
                           'word_vector_id': [0, 1, 2, 3]})
        df_expected = pd.DataFrame({'token': [7, 42],
                                    'reference_id': [[0], [0, 1, 1]],
                                    'word_vector_id': [[1], [0, 2, 3]]})
        df_result = ag.collect_references_and_word_vectors(df, by='token')
        pd.testing.assert_frame_equal(df_result, df_expected)

    def test_unpack_references_and_word_vectors(self):
        df = pd.DataFrame({'token': [7, 42],
                           'reference_id': [[0], [0, 1, 1]],
                           'word_vector_id': [[1], [0, 2, 3]]})
        df_expected = pd.DataFrame({'token': [42, 7, 42, 42],
                                    'reference_id': [0, 0, 1, 1],
                                    'word_vector_id': [0, 1, 2, 3]})
        df_result = ag.unpack_references_and_word_vectors(df)
        pd.testing.assert_frame_equal(df_result, df_expected)

    def test_gen_ids_for_vectors_and_references(self):
        tokenized_sentences = [['Hello', 'world'], ['42']]
        df_expected = pd.DataFrame({'token': ['Hello', 'world', '42'],
                                    'reference_id': [0, 0, 1],
                                    'word_vector_id': [0, 1, 2]})

        df_result = ag.gen_ids_for_vectors_and_references(tokenized_sentences)
        pd.testing.assert_frame_equal(df_result, df_expected)

    def test_concat_word_vectors(self):
        word_vectors = [torch.tensor([[1, 1], [1, 1]]), torch.tensor([[0, 0]])]
        expected = torch.tensor([[1, 1], [1, 1], [0, 0]])
        result = ag.concat_word_vectors(word_vectors)
        self.assertTrue(torch.equal(expected, result))


if __name__ == '__main__':
    main()
