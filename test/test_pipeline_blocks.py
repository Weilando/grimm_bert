from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from analysis import pipeline_blocks as pb


class TestPipelineBlocks(TestCase):
    @patch('data.corpus_handler.CorpusHandler')
    def test_load_and_preprocess_sentences(self, corpus):
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
    def test_evaluate_clustering(self, corpus):
        """ Should calculate the ARI and AMI for a perfect clustering. """
        corpus.get_tagged_tokens.return_value = pd.DataFrame({
            'token': ['a', 'b', 'a'], 'sense': ['a0', 'b0', 'a1']})
        flat_dict_senses = pd.DataFrame({
            'word_vector_id': [0, 1, 2], 'sense': ['a0', 'b0', 'a1']})

        with self.assertLogs(level="INFO") as logs:
            stats = pb.evaluate_clustering(corpus, flat_dict_senses)

        self.assertEqual({'ari': 1.0}, stats)
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(logs.records[0].getMessage(), "ARI: 1.0")
