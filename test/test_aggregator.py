from unittest import main, TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

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

    def test_extract_int_senses_from_df(self):
        """ Should generate a unique integer labels per sense. """
        df = pd.DataFrame({'sense': ['a', 'b', 'c', 'a', 'd']})
        id_senses_exp = np.array([0, 1, 2, 0, 3])
        id_senses_res = ag.extract_int_senses_from_df(df)
        np.testing.assert_array_equal(id_senses_exp, id_senses_res)

    def test_extract_int_senses_from_list(self):
        """ Should generate a unique integer labels per sense. """
        senses = ['a', 'b', 'c', 'a', 'd']
        id_senses_exp = np.array([0, 1, 2, 0, 3])
        id_senses_res = ag.extract_int_senses_from_list(senses)
        np.testing.assert_array_equal(id_senses_exp, id_senses_res)

    def test_count_total_and_unique(self):
        """ Should generate a dict with correct statistics. """
        df = pd.DataFrame({'a': [0, 1, 1, 0, 2],
                           'b': [0, 0, 0, 1, 1]})
        expected = {'total_a_count': 5, 'unique_a_count': 3}
        self.assertEqual(expected, ag.count_total_and_unique(df, 'a'))

    def test_count_unique_senses_per_token(self):
        """ Should group the DataFrame by 'token' and count occurrences and
        unique senses. """
        df = pd.DataFrame({'token': ['a', 'b', 'b', 'a', 'b', 'b'],
                           'sense': ['a0', 'b0', 'b1', 'a0', 'b2', 'b1']})
        expected = pd.DataFrame({'token': ['a', 'b'],
                                 'unique_sense_count': [1, 3],
                                 'total_token_count': [2, 4]})
        pd.testing.assert_frame_equal(expected,
                                      ag.count_unique_senses_per_token(df))

    def test_add_sense_counts_to_id_map(self):
        """ Should append 'unique_sense_count' and 'total_token_count' from
        'sense_counts' to 'id_map'. """
        id_map = pd.DataFrame({'token': ['a', 'b'],
                               'reference_id': [[0], [0, 1, 1]],
                               'word_vector_id': [[1], [0, 2, 3]]})
        sense_counts = pd.DataFrame({'token': ['a', 'b'],
                                     'unique_sense_count': [1, 2],
                                     'total_token_count': [2, 4]})
        expected_id_map = pd.DataFrame({'token': ['a', 'b'],
                                        'reference_id': [[0], [0, 1, 1]],
                                        'word_vector_id': [[1], [0, 2, 3]],
                                        'unique_sense_count': [1, 2],
                                        'total_token_count': [2, 4]})
        result_id_map = ag.add_sense_counts_to_id_map(id_map, sense_counts)
        pd.testing.assert_frame_equal(expected_id_map, result_id_map)

    def test_count_tokens_per_sense_count(self):
        """ Should aggregate the correct number of total and unique tokens per
        sense count. """
        sense_counts = pd.DataFrame({'token': ['a', 'b', 'c'],
                                     'unique_sense_count': [3, 4, 3],
                                     'total_token_count': [2, 5, 4]})
        expected = pd.DataFrame({'unique_sense_count': [3, 4],
                                 'unique_token_count': [2, 1],
                                 'total_token_count': [6, 5]})
        result = ag.count_tokens_per_sense_count(sense_counts)
        pd.testing.assert_frame_equal(expected, result)

    def test_count_monosemous_and_polysemous_tokens(self):
        """ Should count polysemous and monosemous tokens correctly. """
        sense_counts = pd.DataFrame({'token': ['a', 'b', 'c', 'd', 'e'],
                                     'unique_sense_count': [3, 1, 5, 1, 4],
                                     'total_token_count': [5, 5, 5, 5, 5]})
        expected = {'total_monosemous_token_count': 10,
                    'unique_monosemous_token_count': 2,
                    'total_polysemous_token_count': 15,
                    'unique_polysemous_token_count': 3}
        result = ag.count_monosemous_and_polysemous_tokens(sense_counts)
        self.assertEqual(expected, result)

    @patch('data.corpus_handler.CorpusHandler')
    def test_calc_corpus_statistics(self, corpus):
        """ Should count the unique senses per lower cased token from 'corpus'
        and add the counts to id_map. """
        corpus.get_tagged_tokens.return_value = pd.DataFrame({
            'token': ['a', 'b', 'a', '.', 'b', '.'],
            'sense': ['a0', 'b0', 'a1', '.0', 'b0', '.0'],
            'tagged_sense': [True, True, True, True, True, False]})
        corpus.get_sentences.return_value = pd.DataFrame({
            'sentence': [['a', 'b', 'a', '.'], ['b', '.']]})

        result_stats = ag.calc_corpus_statistics(corpus)
        expected_stats = {'sentence_count': 2,
                          'tagged_sense_count': 5,
                          'total_sense_count': 6,
                          'unique_sense_count': 4,
                          'total_token_count': 6,
                          'unique_token_count': 3,
                          'total_monosemous_token_count': 4,
                          'unique_monosemous_token_count': 2,
                          'total_polysemous_token_count': 2,
                          'unique_polysemous_token_count': 1}
        self.assertEqual(expected_stats, result_stats)


if __name__ == '__main__':
    main()
