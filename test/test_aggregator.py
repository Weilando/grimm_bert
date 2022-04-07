from unittest import main, TestCase

import pandas as pd
import torch

from analysis import aggregator as ag


class TestAggregator(TestCase):
    def test_collect_references_and_word_vectors(self):
        """ Should group the DataFrame by the tokens and collect reference_ids
        and word_vector_ids in lists. """
        df = pd.DataFrame({'token': [42, 7, 42, 42],
                           'reference_id': [0, 0, 1, 1],
                           'word_vector_id': [0, 1, 2, 3]})
        df_expected = pd.DataFrame({'token': [7, 42],
                                    'reference_id': [[0], [0, 1, 1]],
                                    'word_vector_id': [[1], [0, 2, 3]]})
        df_result = ag.collect_references_and_word_vectors(df, by='token')
        pd.testing.assert_frame_equal(df_result, df_expected)

    def test_unpack_per_word_vector(self):
        """ Should undo the group operation and unpack the lists. """
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
        """ Should generate unique integer ids for word vectors and references
        for each token. """
        tokenized_sentences = [['Hello', 'world'], ['42']]
        df_expected = pd.DataFrame({'token': ['Hello', 'world', '42'],
                                    'reference_id': [0, 0, 1],
                                    'word_vector_id': [0, 1, 2]})

        df_result = ag.gen_ids_for_vectors_and_references(tokenized_sentences)
        pd.testing.assert_frame_equal(df_result, df_expected)

    def test_concat_word_vectors(self):
        """ Should generate a matrix with input vectors as rows. """
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

    def test_count_total_and_unique(self):
        """ Should generate a dict with correct statistics. """
        df = pd.DataFrame({'a': [0, 1, 1, 0, 2],
                           'b': [0, 0, 0, 1, 1]})
        expected = {'total_a_count': 5, 'unique_a_count': 3}
        self.assertEqual(expected, ag.count_total_and_unique(df, 'a'))

    def test_count_unique_senses_per_token(self):
        df = pd.DataFrame({'token': ['a', 'b', 'b', 'a', 'b', 'b'],
                           'sense': ['a0', 'b0', 'b1', 'a0', 'b2', 'b1']})
        expected = pd.DataFrame({'token': ['a', 'b'], 'n_senses': [1, 3]})
        pd.testing.assert_frame_equal(expected,
                                      ag.count_unique_senses_per_token(df))

    def test_add_sense_counts_to_id_map(self):
        id_map = pd.DataFrame({'token': [7, 42],
                               'reference_id': [[0], [0, 1, 1]],
                               'word_vector_id': [[1], [0, 2, 3]]})
        sense_counts = pd.DataFrame({'token': [7, 42],
                                     'n_senses': [1, 2]})
        expected_id_map = pd.DataFrame({'token': [7, 42],
                                        'reference_id': [[0], [0, 1, 1]],
                                        'word_vector_id': [[1], [0, 2, 3]],
                                        'n_senses': [1, 2]})
        result_id_map = ag.add_sense_counts_to_id_map(id_map, sense_counts)
        pd.testing.assert_frame_equal(expected_id_map, result_id_map)


if __name__ == '__main__':
    main()
