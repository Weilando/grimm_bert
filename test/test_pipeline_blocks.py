from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from analysis import pipeline_blocks as pb


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

    @patch('data.corpus_handler.CorpusHandler')
    def test_add_sense_counts_to_id_map(self, corpus):
        """ Should count the unique senses per lower cased token from 'corpus'
        and add the counts to id_map. """
        corpus.get_tagged_tokens.return_value = pd.DataFrame({
            'token': ['A', 'b', 'a', 'b', '.'],
            'sense': ['a0', 'b0', 'a0', 'b1', '.0']})
        id_map = pd.DataFrame({
            'token': ['a', 'b', '.'],
            'word_vector_id': [[0, 2], [1, 3], [4]],
            'reference_id': [[0, 0], [0, 0], [0]]})

        with self.assertLogs(level="INFO") as captured_logs:
            result = pb.add_sense_counts_to_id_map(corpus, id_map)

            expected = pd.DataFrame({
                'token': ['a', 'b', '.'],
                'word_vector_id': [[0, 2], [1, 3], [4]],
                'reference_id': [[0, 0], [0, 0], [0]],
                'n_senses': [1, 2, 1]})
        pd.testing.assert_frame_equal(expected, result)
        self.assertEqual(len(captured_logs.records), 1)
        self.assertIn("Loaded ground truth number of senses per token.",
                      captured_logs.output[0])

    @patch('data.corpus_handler.CorpusHandler')
    def test_evaluate_clustering(self, corpus):
        """ Should calculate the ARI for a perfect clustering. """
        corpus.get_tagged_tokens.return_value = pd.DataFrame({
            'token': ['a', 'b', 'a'], 'sense': ['a0', 'b0', 'a1']})
        flat_dict_senses = pd.DataFrame({
            'word_vector_id': [0, 1, 2], 'sense': ['a0', 'b0', 'a1']})

        with self.assertLogs(level="INFO") as logs:
            stats = pb.evaluate_clustering(corpus, flat_dict_senses)

        self.assertEqual({'ari': 1.0}, stats)
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(logs.records[0].getMessage(), "ARI: 1.0")
