import os
from tempfile import TemporaryDirectory
from unittest import main, TestCase

import numpy as np
import pandas as pd

import data.file_handler as fh


class TestFileHandler(TestCase):
    def test_add_and_get_absolute_path(self):
        """ Should return the correct absolute result path. """
        with TemporaryDirectory() as tmp_dir_name:
            os.chdir(tmp_dir_name)
            expected_path = os.path.join(os.getcwd(), 'results')
            os.mkdir(expected_path)

            result_path = fh.add_and_get_abs_path("results")

            self.assertEqual(expected_path, result_path)
            self.assertTrue(os.path.exists(expected_path))

    def test_add_and_get_absolute_path_with_mkdir(self):
        """ Should return the correct absolute path and add the directory. """
        with TemporaryDirectory() as tmp_dir_name:
            os.chdir(tmp_dir_name)
            expected_path = os.path.join(os.getcwd(), 'results')

            result_path = fh.add_and_get_abs_path("results")

            self.assertEqual(expected_path, result_path)
            self.assertTrue(os.path.exists(expected_path))

    def test_file_does_exist(self):
        with TemporaryDirectory() as tmp_dir_name:
            fh.save_matrix(tmp_dir_name, 'm.npz', np.ones(1))
            self.assertTrue(fh.does_file_exist(tmp_dir_name, 'm.npz'))

    def test_file_does_not_exist(self):
        with TemporaryDirectory() as tmp_dir_name:
            self.assertFalse(fh.does_file_exist(tmp_dir_name, 'm.npz'))

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

    def test_save_and_load_stats(self):
        """ Should save a dict into a json-file and load it afterwards. """
        stats = {'ari': 0.5}
        file_name = 'stats.json'

        with TemporaryDirectory() as tmp_dir_name:
            fh.save_stats(tmp_dir_name, file_name, stats)
            reconstructed_stats = fh.load_stats(tmp_dir_name, file_name)
            self.assertEqual(stats, reconstructed_stats)


if __name__ == '__main__':
    main()
