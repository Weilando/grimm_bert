from tempfile import TemporaryDirectory
from unittest import TestCase, main

import pandas as pd

import data.toy_preprocessor as tp
from data.corpus_handler import CorpusName, CorpusHandler


class TestCorpusName(TestCase):
    def test_get_values(self):
        expected = ['SemCor', 'SemEval2007', 'SemEval2013', 'SemEval2015',
                    'Senseval2', 'Senseval3', 'Toy']
        self.assertEqual(expected, CorpusName.get_values())

    def test_is_wsdeval_name(self):
        self.assertFalse(CorpusName.TOY.is_wsdeval_name)
        self.assertTrue(CorpusName.SEMCOR.is_wsdeval_name)
        self.assertTrue(CorpusName.SENSEVAL2.is_wsdeval_name)
        self.assertTrue(CorpusName.SENSEVAL3.is_wsdeval_name)
        self.assertTrue(CorpusName.SEMEVAL07.is_wsdeval_name)
        self.assertTrue(CorpusName.SEMEVAL13.is_wsdeval_name)
        self.assertTrue(CorpusName.SEMEVAL15.is_wsdeval_name)


class TestToyPreprocessor(TestCase):
    def test_flatten_list(self):
        nested_list = [['a', 'b'], ['c']]
        self.assertEqual(['a', 'b', 'c'], tp.flatten_list(nested_list))


class TestCorpusHandler(TestCase):
    def test_set_up_toy_corpus_handler(self):
        with TemporaryDirectory() as tmp_dir_name:
            corpus = CorpusHandler(CorpusName.TOY, tmp_dir_name)
            self.assertEqual(CorpusName.TOY, corpus.corpus_name)
            self.assertEqual(tmp_dir_name, corpus.corpus_path)
            self.assertEqual("toy-sentences.pkl", corpus.sentences_name)
            self.assertEqual("toy-tagged_tokens.pkl",
                             corpus.tagged_tokens_name)

    def test_access_toy_via_corpus_handler(self):
        with TemporaryDirectory() as tmp_dir:
            toy_preprocessor = tp.ToyPreprocessor(corpus_cache_path=tmp_dir)
            toy_preprocessor.cache_dataset()

            corpus = CorpusHandler(CorpusName.TOY, tmp_dir)

            pd.testing.assert_frame_equal(toy_preprocessor.get_sentences(),
                                          corpus.get_sentences())
            self.assertEqual(tp.SENTENCES, corpus.get_sentences_as_list())
            pd.testing.assert_frame_equal(toy_preprocessor.get_tagged_tokens(),
                                          corpus.get_tagged_tokens())


if __name__ == '__main__':
    main()
