from tempfile import TemporaryDirectory
from unittest import TestCase, main

import pandas as pd

import data.toy_preprocessor as tp
from data.corpus_handler import CorpusName, CorpusHandler


class TestCorpusName(TestCase):
    def test_get_names(self):
        expected = ['toy', 'semcor', 'semeval2007', 'semeval2013',
                    'semeval2015', 'senseval2', 'senseval3']
        self.assertEqual(expected, CorpusName.get_names())

    def test_is_wsdeval_name(self):
        self.assertTrue(CorpusName.SENSEVAL2.is_wsdeval_name)
        self.assertTrue(CorpusName.SENSEVAL3.is_wsdeval_name)
        self.assertTrue(CorpusName.SEMEVAL07.is_wsdeval_name)
        self.assertTrue(CorpusName.SEMEVAL13.is_wsdeval_name)
        self.assertTrue(CorpusName.SEMEVAL15.is_wsdeval_name)
        self.assertFalse(CorpusName.TOY.is_wsdeval_name)
        self.assertFalse(CorpusName.SEMCOR.is_wsdeval_name)


class TestCorpusHandler(TestCase):
    def test_set_up_toy(self):
        with TemporaryDirectory() as tmp_dir_name:
            corpus = CorpusHandler(CorpusName.TOY, tmp_dir_name)
            self.assertEqual(CorpusName.TOY, corpus.corpus_name)
            self.assertEqual(tmp_dir_name, corpus.corpus_path)
            self.assertEqual("toy-sentences.pkl", corpus.sentences_name)
            self.assertEqual("toy-tagged_tokens.pkl",
                             corpus.tagged_tokens_name)

    def test_get_sentences(self):
        with TemporaryDirectory() as tmp_dir:
            toy_preprocessor = tp.ToyPreprocessor(corpus_cache_path=tmp_dir)
            toy_preprocessor.cache_dataset()

            expected = pd.DataFrame({'sentence': tp.SENTENCES})
            corpus = CorpusHandler(CorpusName.TOY, tmp_dir)
            pd.testing.assert_frame_equal(expected, corpus.get_sentences())

    def test_get_sentences_as_list(self):
        with TemporaryDirectory() as tmp_dir:
            toy_preprocessor = tp.ToyPreprocessor(corpus_cache_path=tmp_dir)
            toy_preprocessor.cache_dataset()

            corpus = CorpusHandler(CorpusName.TOY, tmp_dir)
            self.assertEqual(tp.SENTENCES, corpus.get_sentences_as_list())

    def test_get_tagged_tokens(self):
        with TemporaryDirectory() as tmp_dir:
            toy_preprocessor = tp.ToyPreprocessor(corpus_cache_path=tmp_dir)
            toy_preprocessor.cache_dataset()

            expected = pd.DataFrame({'token': tp.TOKENS, 'sense': tp.SENSES})
            corpus = CorpusHandler(CorpusName.TOY, tmp_dir)
            pd.testing.assert_frame_equal(expected, corpus.get_tagged_tokens())


if __name__ == '__main__':
    main()
