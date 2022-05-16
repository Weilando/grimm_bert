from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

import aggregation.pipeline_blocks as pb
from clustering.metric_name import MetricName
from data.corpus_handler import CorpusHandler
from data.corpus_name import CorpusName


class TestPipelineBlocks(TestCase):
    @patch('data.corpus_handler.CorpusHandler')
    def test_load_and_preprocess_sentences(self, corpus):
        """ Should load all sentences from 'corpus', lower case each token and
        add special tokens per sentence. """
        corpus.get_sentences_as_list.return_value = [['hello', 'world', '!'],
                                                     ['hi', '.']]

        with self.assertLogs(level="INFO") as captured_logs:
            sentences = pb.load_and_preprocess_sentences(corpus)

            expected = [['[CLS]', 'hello', 'world', '!', '[SEP]'],
                        ['[CLS]', 'hi', '.', '[SEP]']]
        self.assertEqual(expected, sentences)
        self.assertEqual(len(captured_logs.records), 1)
        self.assertIn("Lower cased sentences and added special tokens.",
                      captured_logs.output[0])

    def test_add_sense_counts_to_id_map(self):
        """ Should count the unique senses per lower cased token from
        'tagged_tokens' and add the counts to id_map. """
        tagged_tokens = pd.DataFrame({
            'token': ['a', 'b', 'a', 'b', '.'],
            'sense': ['a0', 'b0', 'a0', 'b1', '.0'],
            'tagged_sense': [True, True, True, True, True]})
        id_map = pd.DataFrame({'token': ['a', 'b', '.'],
                               'token_id': [[0, 2], [1, 3], [4]],
                               'sentence_id': [[0, 0], [0, 0], [0]]})
        expected = pd.DataFrame({'token': ['a', 'b', '.'],
                                 'token_id': [[0, 2], [1, 3], [4]],
                                 'sentence_id': [[0, 0], [0, 0], [0]],
                                 'unique_sense_count': [1, 2, 1],
                                 'total_token_count': [2, 2, 1]})

        with self.assertLogs(level="INFO") as captured_logs:
            result = pb.add_sense_counts_to_id_map(tagged_tokens, id_map)

        pd.testing.assert_frame_equal(expected, result)
        self.assertEqual(len(captured_logs.records), 1)
        self.assertIn("Loaded ground truth number of senses per token.",
                      captured_logs.output[0])

    @patch('data.file_handler.does_file_exist', return_value=True)
    def test_does_word_vector_cache_exist(self, does_file_exist):
        """ Should return True, as both files exist. """
        self.assertTrue(pb.does_word_vector_cache_exist(
            '/path', 'word_vec_file', 'raw_id_map_file'))
        self.assertEqual(2, does_file_exist.call_count)
        does_file_exist.assert_any_call('/path', 'word_vec_file')
        does_file_exist.assert_any_call('/path', 'raw_id_map_file')

    @patch('data.file_handler.does_file_exist', return_value=False)
    def test_does_word_vector_cache_exist_false(self, does_file_exist):
        """ Should return False, as at least one file is missing. """
        self.assertFalse(pb.does_word_vector_cache_exist(
            '/path', 'word_vec_file', 'raw_id_map_file'))
        does_file_exist.assert_called()

    @patch('aggregation.pipeline_blocks.calculate_word_vectors',
           return_value=(np.ones(3), pd.DataFrame({'a': [42]})))
    def test_get_word_vectors_calculate(self, calculate_word_vectors):
        """ Should calculate the word vectors and id_map from scratch, as no
        cached files exist. """
        with TemporaryDirectory() as tmp_dir:
            corpus = CorpusHandler(CorpusName.TOY, tmp_dir)
            word_vectors, id_map = pb.get_word_vectors(corpus, tmp_dir, tmp_dir)

        np.testing.assert_array_equal(np.ones(3), word_vectors)
        pd.testing.assert_frame_equal(pd.DataFrame({'a': [42]}), id_map)
        calculate_word_vectors.assert_called()

    def test_calc_ari_for_tagged_senses(self):
        """ Should calculate the ARI for a perfect clustering. Should only
        consider the tagged token and therefore a perfect score. """
        tagged_tokens = pd.DataFrame({'token': ['a', 'a'],
                                      'sense': ['a0', 'a0'],
                                      'tagged_sense': [True, False]})
        flat_dict_senses = pd.DataFrame({
            'token_id': [0, 1], 'sense': ['a0', 'a1']})

        with self.assertLogs(level="INFO") as logs:
            stats = pb.calc_ari(tagged_tokens, flat_dict_senses)

        self.assertEqual({'ari': 1.0}, stats)
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(logs.records[0].getMessage(), "ARI: 1.0")

    def test_calc_ari_per_token(self):
        """ Should add a column with one ARI per token and an indicator for
        tokens with completely tagged senses. """
        tagged_tokens = pd.DataFrame({
            'token': ['a', 'b', 'a', 'b', '.'],
            'sense': ['a0', 'b0', 'a0', 'b1', '.0'],
            'tagged_sense': [True, True, True, False, False]})
        dictionary = pd.DataFrame({
            'token': ['a', 'b', '.'],
            'token_id': [[0, 2], [1, 3], [4]],
            'sense': [['a0', 'a1'], ['b0', 'b1'], ['.0']]})
        expected = pd.DataFrame({
            'token': ['a', 'b', '.'],
            'token_id': [[0, 2], [1, 3], [4]],
            'sense': [['a0', 'a1'], ['b0', 'b1'], ['.0']],
            'ari': [0.0, 1.0, 1.0],
            'tagged_token': [True, False, False]})

        result = pb.calc_ari_per_token(tagged_tokens, dictionary)
        pd.testing.assert_frame_equal(expected, result)

    def test_calc_silhouette_score_per_token(self):
        """ Should add a column with one Silhouette Coefficient per token and
        NaN if too few or many labels are given to calculate the score. """
        vectors = np.array([[5, 2], [9, 0], [5, 3], [1, 0], [0, 1], [7, 1]])
        dictionary = pd.DataFrame({
            'token': ['a', 'b', '.'],
            'token_id': [[0, 2], [1, 3, 4], [5]],
            'sense': [['a0', 'a1'], ['b0', 'b1', 'b1'], ['.0']]})
        expected = pd.DataFrame({
            'token': ['a', 'b', '.'],
            'token_id': [[0, 2], [1, 3, 4], [5]],
            'sense': [['a0', 'a1'], ['b0', 'b1', 'b1'], ['.0']],
            'silhouette_score': [np.NaN, 0.5557, np.NaN]})

        result = pb.calc_silhouette_score_per_token(vectors, dictionary,
                                                    MetricName.EUCLIDEAN)
        pd.testing.assert_frame_equal(expected, result, atol=1e-4)
