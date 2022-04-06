from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from analysis import pipeline_blocks as pb
from data import toy_preprocessor as tp
from data.corpus_name import CorpusName


class TestPipelineBlocks(TestCase):
    def test_load_and_preprocess_sentences(self):
        with TemporaryDirectory() as tmp_dir:
            toy_preprocessor = tp.ToyPreprocessor(corpus_cache_path=tmp_dir)
            toy_preprocessor.cache_dataset()

            with self.assertLogs(level="INFO") as captured_logs:
                sentences = pb.load_and_preprocess_sentences(CorpusName.TOY,
                                                             tmp_dir)

            expected = [['[CLS]', 'he', 'wears', 'a', 'watch', '.', '[SEP]'],
                        ['[CLS]', 'she', 'glances', 'at', 'the', 'watch',
                         'often', '.', '[SEP]'],
                        ['[CLS]', 'he', 'wants', 'to', 'watch', 'the', 'soccer',
                         'match', '.', '[SEP]'],
                        ['[CLS]', 'we', 'watch', 'movies', 'and', 'eat',
                         'popcorn', '.', '[SEP]']]
            self.assertEqual(expected, sentences)
            self.assertEqual(len(captured_logs.records), 1)
            self.assertIn("Lower cased sentences and added special tokens.",
                          captured_logs.output[0])

    def test_evaluate_with_ari(self):
        with TemporaryDirectory() as tmp_dir:
            toy_preprocessor = tp.ToyPreprocessor(corpus_cache_path=tmp_dir)
            toy_preprocessor.cache_dataset()

            flat_dict_senses = pd.DataFrame(
                {'word_vector_id': range(len(tp.TOKENS)),
                 'sense': tp.SENSES})

            with self.assertLogs(level="INFO") as captured_logs:
                stats = pb.evaluate_with_ari(CorpusName.TOY, tmp_dir,
                                             flat_dict_senses)

            self.assertEqual({'ari': 1.0}, stats)
            self.assertEqual(len(captured_logs.records), 1)
            self.assertEqual(captured_logs.records[0].getMessage(), "ARI: 1.0")
