import json
import os
from tempfile import TemporaryDirectory
from unittest import main, TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from data import file_handler as fh


class TestFileHandler(TestCase):
    @patch('data.file_handler.os')
    def test_add_and_get_absolute_path(self, mocked_os):
        """ Should return the correct absolute result path. """
        expected_path = "//root/results"

        mocked_os.getcwd.return_value = "//root"
        mocked_os.path.exists.return_value = True
        mocked_os.path.join.return_value = expected_path

        result_path = fh.add_and_get_abs_path("results")

        self.assertEqual(expected_path, result_path)
        mocked_os.getcwd.assert_called_once()
        mocked_os.path.exists.assert_called_once_with(expected_path)
        mocked_os.path.join.called_once_with("//root", "results")
        mocked_os.mkdir.assert_not_called()

    @patch('data.file_handler.os')
    def test_add_and_get_absolute_path_with_mkdir(self, mocked_os):
        """ Should return the correct absolute path and add the directory. """
        expected_path = "//root/results"

        mocked_os.getcwd.return_value = "//root"
        mocked_os.path.exists.return_value = False
        mocked_os.path.join.return_value = expected_path

        result_path = fh.add_and_get_abs_path("results")

        self.assertEqual(expected_path, result_path)
        mocked_os.getcwd.assert_called_once()
        mocked_os.path.exists.assert_called_once_with(expected_path)
        mocked_os.path.join.called_once_with("//root", "results")
        mocked_os.mkdir.assert_called_once_with(expected_path)

    def test_file_does_exist(self):
        with TemporaryDirectory() as tmp_dir_name:
            fh.save_matrix(tmp_dir_name, 'm.npz', np.ones(1))
            self.assertTrue(fh.does_file_exist(tmp_dir_name, 'm.npz'))

    def test_file_does_not_exist(self):
        with TemporaryDirectory() as tmp_dir_name:
            self.assertFalse(fh.does_file_exist(tmp_dir_name, 'm.npz'))

    def test_gen_dictionary_file_name(self):
        self.assertEqual('cor-linkage_single-dist_0.12345-dictionary.pkl',
                         fh.gen_dictionary_file_name('cor', 'single', 0.12345))

    def test_gen_raw_id_map_file_name(self):
        self.assertEqual('corpus_name-raw_id_map.pkl',
                         fh.gen_raw_id_map_file_name('corpus_name'))

    def test_gen_sentences_file_name(self):
        self.assertEqual('corpus_name-sentences.pkl',
                         fh.gen_sentences_file_name('corpus_name'))

    def test_gen_stats_file_name(self):
        self.assertEqual('cor-linkage_average-dist_0.12345-stats.json',
                         fh.gen_stats_file_name('cor', 'average', 0.12345))

    def test_gen_tagged_tokens_file_name(self):
        self.assertEqual('corpus_name-tagged_tokens.pkl',
                         fh.gen_tagged_tokens_file_name('corpus_name'))

    def test_gen_word_vec_file_name(self):
        self.assertEqual('corpus_name-word_vectors.npz',
                         fh.gen_word_vec_file_name('corpus_name'))

    def test_save_and_load_df(self):
        """ Should save a DataFrame into a pkl-file and load it afterwards. """
        df = pd.DataFrame({'col': [42, 7]})
        file_name = 'df.pkl'

        with TemporaryDirectory() as tmp_dir_name:
            result_file_path = os.path.join(tmp_dir_name, file_name)

            fh.save_df(tmp_dir_name, file_name, df)
            reconstructed_df = fh.load_df(tmp_dir_name, file_name)

            self.assertTrue(os.path.exists(result_file_path))
            self.assertTrue(os.path.isfile(result_file_path))
            pd.testing.assert_frame_equal(df, reconstructed_df)

    def test_save_and_load_matrix(self):
        """ Should save a matrix into a npz-file and load it afterwards. """
        matrix = np.eye(2)
        file_name = 'matrix.npz'

        with TemporaryDirectory() as tmp_dir_name:
            result_file_path = os.path.join(tmp_dir_name, file_name)

            fh.save_matrix(tmp_dir_name, file_name, matrix)
            reconstructed_matrix = fh.load_matrix(tmp_dir_name, file_name)

            self.assertTrue(os.path.exists(result_file_path))
            self.assertTrue(os.path.isfile(result_file_path))
            np.testing.assert_array_equal(matrix, reconstructed_matrix)

    def test_save_stats(self):
        """ Should save a dict with statistics into a json-file. """
        stats = {'ari': 0.5}
        file_name = 'stats.json'

        with TemporaryDirectory() as tmp_dir_name:
            fh.save_stats(tmp_dir_name, file_name, stats)

            with open(os.path.join(tmp_dir_name, file_name), 'r') as stats_file:
                reconstructed_stats = json.load(stats_file)
                self.assertEqual(stats, reconstructed_stats)


if __name__ == '__main__':
    main()
