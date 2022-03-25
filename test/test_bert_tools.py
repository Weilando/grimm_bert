import os.path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from model.character_cnn_utils import CharacterIndexer

from analysis import bert_tools as bt


class TestBertTools(TestCase):
    def test_gen_model_cache_location(self):
        with TemporaryDirectory() as tmp_dir_name:
            self.assertEqual(bt.gen_model_cache_location(tmp_dir_name, 'name'),
                             os.path.join(tmp_dir_name, 'name'))

    @patch('transformers.BertTokenizer')
    def test_tokenize_text(self, mock_tokenizer):
        mock_tokenizer.basic_tokenizer.tokenize.return_value = ['Hi', '!']
        expected = ['[CLS]', 'Hi', '!', '[SEP]']
        self.assertEqual(expected, bt.tokenize_text('Hi!', mock_tokenizer))
        mock_tokenizer.basic_tokenizer.tokenize.assert_called_once_with('Hi!')

    def test_encode_text(self):
        indexer = CharacterIndexer()
        res_encoding = bt.encode_text(['[CLS]', 'Hi', '!', '[SEP]'], indexer)
        self.assertEqual((1, 4, 50), res_encoding.size())
        self.assertEqual(torch.int64, res_encoding.dtype)

    @patch('transformers.BertModel')
    def test_calc_word_vectors(self, mock_model):
        mock_model.return_value = tuple(torch.ones((1, 3, 2)))
        res = bt.calc_word_vectors(torch.ones(1), mock_model)
        self.assertTrue(torch.equal(res, torch.ones((3, 2))))
        mock_model.assert_called_once_with(torch.ones(1))

    @patch('transformers.BertModel')
    @patch('transformers.BertTokenizer')
    def test_parse_sentences(self, mock_model, mock_tokenizer):
        mock_model.return_value = tuple(torch.ones((1, 3, 2)))
        mock_tokenizer.basic_tokenizer.tokenize.return_value = ['Hello']
        indexer = CharacterIndexer()

        word_vectors_res, id_map_res = bt.parse_sentences(
            ['Hello'], mock_tokenizer, indexer, mock_model)
        word_vectors_exp = np.ones((3, 2))
        id_map_exp = pd.DataFrame({'token': ['[CLS]', 'Hello', '[SEP]'],
                                   'reference_id': [0, 0, 0],
                                   'word_vector_id': [0, 1, 2]})

        np.testing.assert_array_equal(word_vectors_res, word_vectors_exp)
        pd.testing.assert_frame_equal(id_map_res, id_map_exp)
        mock_model.assert_called_once()
        mock_tokenizer.basic_tokenizer.tokenize.assert_called_once()


if __name__ == '__main__':
    main()
