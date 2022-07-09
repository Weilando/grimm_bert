import os
from tempfile import TemporaryDirectory, NamedTemporaryFile
from unittest import TestCase, main

import pandas as pd
from transformers import BertTokenizer

import data.raw_text_preprocessor as rp

TOY_RAW_TEXT_CORPUS = "This document is a summary.\n How does it work? "
TOY_RAW_TEXT_CORPUS_LINES = ["This document is a summary.", "How does it work?"]
TOY_RAW_TEXT_CORPUS_TOKENS = [["this", "document", "is", "a", "summary", "."],
                              ["how", "does", "it", "work", "?"]]


class TestRawTextPreprocessor(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = BertTokenizer.from_pretrained(
            './model_cache/bert-base-uncased')

    def test_read_and_strip_lines(self):
        with NamedTemporaryFile(suffix='txt') as txt_file:
            txt_file.write(TOY_RAW_TEXT_CORPUS.encode("utf-8"))
            txt_file.seek(0)

            result_lines = rp.read_and_strip_lines(txt_file.name)
            self.assertEqual(result_lines, TOY_RAW_TEXT_CORPUS_LINES)

    def test_tokenize_lines(self):
        self.assertEqual(
            rp.tokenize_lines(TOY_RAW_TEXT_CORPUS_LINES, self.tokenizer),
            TOY_RAW_TEXT_CORPUS_TOKENS)

    def test_get_sentences(self):
        with TemporaryDirectory() as tmp_dir:
            os.chdir(tmp_dir)
            preprocessor = rp.RawTextPreprocessor(
                TOY_RAW_TEXT_CORPUS_LINES, self.tokenizer,
                corpus_cache_path=tmp_dir)
            sentences = preprocessor.get_sentences()
        expected_sentences = pd.DataFrame({
            'sentence': [['this', 'document', 'is', 'a', 'summary', '.'],
                         ['how', 'does', 'it', 'work', '?']]})
        pd.testing.assert_frame_equal(expected_sentences, sentences)

    def test_get_tagged_tokens(self):
        with TemporaryDirectory() as tmp_dir:
            os.chdir(tmp_dir)
            preprocessor = rp.RawTextPreprocessor(
                TOY_RAW_TEXT_CORPUS_LINES, self.tokenizer,
                corpus_cache_path=tmp_dir)
            tokens = preprocessor.get_tagged_tokens()

        expected_tokens = pd.DataFrame({
            'token': ['this', 'document', 'is', 'a', 'summary', '.', 'how',
                      'does', 'it', 'work', '?'],
            'sense': ['this_SENSE', 'document_SENSE', 'is_SENSE', 'a_SENSE',
                      'summary_SENSE', '._SENSE', 'how_SENSE', 'does_SENSE',
                      'it_SENSE', 'work_SENSE', '?_SENSE'],
            'tagged_sense': [False, False, False, False, False, False, False,
                             False, False, False, False]})
        pd.testing.assert_frame_equal(expected_tokens, tokens)


if __name__ == '__main__':
    main()
