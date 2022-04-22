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

    def test_lower_tokens(self):
        self.assertEqual(['hello', '!'], bt.lower_tokens(['HelLo', '!']))

    def test_add_special_tokens(self):
        expected = ['[CLS]', 'hi', '!', '[SEP]']
        self.assertEqual(expected, bt.add_special_tokens(['hi', '!']))

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

    def test_concat_word_vectors(self):
        """ Should generate a matrix with input vectors as rows. """
        word_vectors = [torch.tensor([[1, 1], [1, 1]]), torch.tensor([[0, 0]])]
        expected = torch.tensor([[1, 1], [1, 1], [0, 0]])
        result = bt.concat_word_vectors(word_vectors)
        self.assertTrue(torch.equal(expected, result))

    def test_strip_tokenized_sentences(self):
        tokenized_sentences = [['[CLS]', 'hi', '[SEP]'],
                               ['[CLS]', 'hello', 'world', '[SEP]']]
        expected = [['hi', ], ['hello', 'world']]
        result = bt.strip_each(tokenized_sentences)
        self.assertEqual(expected, result)

    def test_strip_word_vector_matrix(self):
        word_vector_matrices = [torch.tensor([[0], [1], [2]]),
                                torch.tensor([[3], [4], [5], [6]])]
        expected = [torch.tensor([[1]]), torch.tensor([[4], [5]])]
        result = bt.strip_each(word_vector_matrices)
        self.assertTrue(torch.equal(expected[0], result[0]))
        self.assertTrue(torch.equal(expected[1], result[1]))

    def test_add_special_tokens_to_each(self):
        expected = [['[CLS]', 'a', 'b', '[SEP]'], ['[CLS]', 'c', '[SEP]']]
        result = bt.add_special_tokens_to_each([['a', 'b'], ['c']])
        self.assertEqual(expected, result)

    def test_lower_sentences(self):
        expected = [['hello'], ['hi', '!']]
        self.assertEqual(expected, bt.lower_sentences([['HelLo'], ['Hi', '!']]))

    @patch('transformers.BertModel')
    def test_embed_sentences(self, mock_model):
        mock_model.side_effect = [tuple(torch.ones((1, 5, 2))),
                                  tuple(torch.ones((1, 3, 2)))]
        indexer = CharacterIndexer()

        word_vectors_res, id_map_res = bt.embed_sentences(
            [['[CLS]', 'a', 'b', 'c', '[SEP]'], ['[CLS]', 'd', '[SEP]']],
            indexer, mock_model)
        word_vectors_exp = np.ones((4, 2))  # squeezed and stripped
        id_map_exp = pd.DataFrame({'token': ['a', 'b', 'c', 'd'],
                                   'reference_id': [0, 0, 0, 1],
                                   'word_vector_id': [0, 1, 2, 3]})

        np.testing.assert_array_equal(word_vectors_res, word_vectors_exp)
        pd.testing.assert_frame_equal(id_map_res, id_map_exp)
        self.assertEqual(2, mock_model.call_count)


if __name__ == '__main__':
    main()
