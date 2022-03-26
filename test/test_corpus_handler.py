from tempfile import TemporaryDirectory
from unittest import TestCase, main

import pandas as pd

import data.toy_preprocessor as tp
from data.corpus_handler import CorpusName, CorpusHandler


class TestCorpusName(TestCase):
    def test_get_sentences(self):
        self.assertEqual(['Toy', 'SemCor'], CorpusName.get_names())


class TestCorpusHandler(TestCase):
    def test_set_up_toy(self):
        with TemporaryDirectory() as tmp_dir_name:
            corpus = CorpusHandler(CorpusName.TOY, tmp_dir_name)
            self.assertEqual(CorpusName.TOY, corpus.corpus_name)
            self.assertEqual(tmp_dir_name, corpus.corpus_path)
            self.assertEqual("toy_sentences-df.pkl", corpus.sentences_name)
            self.assertEqual("toy_tagged_tokens-df.pkl",
                             corpus.tagged_tokens_name)

    def test_set_up_semcor(self):
        with TemporaryDirectory() as tmp_dir_name:
            corpus = CorpusHandler(CorpusName.SEMCOR, tmp_dir_name)
            self.assertEqual(CorpusName.SEMCOR, corpus.corpus_name)
            self.assertEqual(tmp_dir_name, corpus.corpus_path)
            self.assertEqual("semcor_sentences-df.pkl", corpus.sentences_name)
            self.assertEqual("semcor_tagged_tokens-df.pkl",
                             corpus.tagged_tokens_name)

    def test_get_sentences(self):
        expected = pd.DataFrame({'sentence': tp.SENTENCES})
        corpus = CorpusHandler(CorpusName.TOY)
        pd.testing.assert_frame_equal(expected, corpus.get_sentences())

    def test_get_sentences_as_list(self):
        corpus = CorpusHandler(CorpusName.TOY)
        self.assertEqual(tp.SENTENCES, corpus.get_sentences_as_list())

    def test_get_tagged_tokens(self):
        expected = pd.DataFrame({'token': tp.TOKENS, 'sense': tp.SENSES})
        corpus = CorpusHandler(CorpusName.TOY)
        pd.testing.assert_frame_equal(expected, corpus.get_tagged_tokens())


if __name__ == '__main__':
    main()
