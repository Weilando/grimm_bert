import os.path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch
from transformers import BatchEncoding

from analysis import bert_tools as bt


class TestBertTools(TestCase):
    def test_gen_model_cache_location(self):
        with TemporaryDirectory() as tmp_dir_name:
            self.assertEqual(bt.gen_model_cache_location(tmp_dir_name, 'name'),
                             os.path.join(tmp_dir_name, 'name'))

    def test_should_cache_model(self):
        """ Should be True, because the cache directory does not exist yet. """
        with TemporaryDirectory() as tmp_dir_name:
            cache_location = bt.gen_model_cache_location(tmp_dir_name, 'model')
            self.assertTrue(bt.should_cache_model(cache_location))

    def test_should_not_cache_model(self):
        """ Should be False, because the cache directory already exists. """
        with TemporaryDirectory() as tmp_dir_name:
            os.mkdir(os.path.join(tmp_dir_name, 'model'))
            cache_location = bt.gen_model_cache_location(tmp_dir_name, 'model')
            self.assertFalse(bt.should_cache_model(cache_location))

    @patch("analysis.bert_tools.logging.info")
    def test_cache_model(self, mock_log_info):
        mock_model = Mock('transformers.BertModel')
        mock_model.save_pretrained = Mock()
        mock_tokenizer = Mock('transformers.BertTokenizer')
        mock_tokenizer.save_pretrained = Mock()
        cache_location = '/cache'
        bt.cache_model(mock_tokenizer, mock_model, cache_location)

        mock_tokenizer.save_pretrained.assert_called_once_with(cache_location)
        mock_model.save_pretrained.assert_called_once_with(cache_location)
        expected_log = "Cached model and tokenizer at /cache."
        mock_log_info.assert_called_once_with(expected_log)

    @patch('transformers.BertTokenizer')
    def test_encode_text(self, mock_tokenizer):
        encoding = BatchEncoding({'input_ids': torch.tensor([101, 42, 103]),
                                  'token_type_ids': torch.zeros(3),
                                  'attention_mask': torch.ones(3)})
        mock_tokenizer.return_value = encoding
        self.assertEqual(bt.encode_text('Hi!', mock_tokenizer), encoding)
        mock_tokenizer.assert_called_once_with('Hi!', return_tensors='pt')

    @patch('transformers.BertTokenizer')
    def test_add_decoded_tokens(self, mock_tokenizer):
        mock_tokenizer.decode.return_value = 'token'
        df = pd.DataFrame({'token': [3, 7]})
        df_exp = pd.DataFrame({'token': [3, 7],
                               'decoded_token': ['token', 'token']})
        df_res = bt.add_decoded_tokens(df, mock_tokenizer)
        pd.testing.assert_frame_equal(df_res, df_exp)
        self.assertEqual(mock_tokenizer.decode.call_count, 2)

    @patch('transformers.BertModel')
    def test_calc_word_vectors(self, mock_model):
        mock_model.return_value = Mock(last_hidden_state=torch.tensor([7]))
        encoding = BatchEncoding({'input_ids': torch.tensor([101, 42, 103]),
                                  'token_type_ids': torch.zeros(3),
                                  'attention_mask': torch.ones(3)})

        res = bt.calc_word_vectors(encoding, mock_model)
        self.assertTrue(torch.equal(res, torch.tensor([7])))
        mock_model.assert_called_once()

    @patch('transformers.BertModel')
    @patch('transformers.BertTokenizer')
    def test_parse_sentences(self, mock_model, mock_tokenizer):
        mock_model.return_value = Mock(last_hidden_state=torch.ones((1, 1, 2)))
        encoding = BatchEncoding({'input_ids': torch.tensor([101, 42, 103]),
                                  'token_type_ids': torch.zeros(3),
                                  'attention_mask': torch.ones(3)})
        mock_tokenizer.return_value = encoding

        word_vectors_res, id_map_res = bt.parse_sentences(['Hello'],
                                                          mock_tokenizer,
                                                          mock_model)
        word_vectors_exp = np.ones((1, 2))
        id_map_exp = pd.DataFrame({'token': [101, 42, 103],
                                   'reference_id': [0, 0, 0],
                                   'word_vector_id': [0, 1, 2]})
        np.testing.assert_array_equal(word_vectors_res, word_vectors_exp)
        pd.testing.assert_frame_equal(id_map_res, id_map_exp)
        mock_model.assert_called_once()
        mock_tokenizer.assert_called_once()


if __name__ == '__main__':
    main()
