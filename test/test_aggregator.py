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

    def test_unpack_per_word_vector(self):
        df = pd.DataFrame({'token': [7, 42],
                           'reference_id': [[0], [0, 1, 1]],
                           'word_vector_id': [[1], [0, 2, 3]]})
        df_expected = pd.DataFrame({'token': [42, 7, 42, 42],
                                    'reference_id': [0, 0, 1, 1],
                                    'word_vector_id': [0, 1, 2, 3]})
        df_result = ag.unpack_per_word_vector(df, ['reference_id',
                                                   'word_vector_id'])
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

    def test_extract_flat_senses(self):
        """ Should unpack and sort senses regarding their word_vector_ids.
        Should drop reference_ids. """
        dictionary = pd.DataFrame({'word_vector_id': [[1, 2], [0]],
                                   'reference_ids': [[0, 1], [1]],
                                   'sense': [['c', 'a'], ['b']]})
        dictionary_exp = pd.DataFrame({'word_vector_id': [0, 1, 2],
                                       'sense': ['b', 'c', 'a']}) \
            .set_index('word_vector_id')
        dictionary_res = ag.extract_flat_senses(dictionary)
        pd.testing.assert_frame_equal(dictionary_exp, dictionary_res)

    def test_extract_int_senses(self):
        """ Should generate a unique integer labels per sense. """
        dictionary = pd.DataFrame({'sense': ['a', 'b', 'c', 'a', 'd']})
        id_senses_exp = [0, 1, 2, 0, 3]
        id_senses_res = ag.extract_int_senses(dictionary)
        self.assertEqual(id_senses_exp, id_senses_res)


if __name__ == '__main__':
    main()
