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

    def test_gen_df_file_name(self):
        self.assertEqual('name-df.pkl', fh.gen_df_file_name('name'))

    def test_gen_matrix_file_name(self):
        self.assertEqual('name-matrix.npz', fh.gen_matrix_file_name('name'))

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

    def test_save_and_load_results(self):
        """ Should save a matrix into a npz-file and load it afterwards. """
        df = pd.DataFrame({'col': [42, 7]})
        matrix = np.eye(2)
        df_file_name = 'name-df.pkl'
        matrix_file_name = 'name-matrix.npz'

        with TemporaryDirectory() as tmp_dir_name:
            df_file_path = os.path.join(tmp_dir_name, df_file_name)
            matrix_file_path = os.path.join(tmp_dir_name, matrix_file_name)

            fh.save_results('name', tmp_dir_name, matrix, df)
            reconstructed_df = fh.load_df(tmp_dir_name, df_file_name)
            reconstructed_matrix = fh.load_matrix(tmp_dir_name,
                                                  matrix_file_name)

            self.assertTrue(os.path.exists(df_file_path))
            self.assertTrue(os.path.exists(matrix_file_path))
            self.assertTrue(os.path.isfile(df_file_path))
            self.assertTrue(os.path.isfile(matrix_file_path))
            pd.testing.assert_frame_equal(df, reconstructed_df)
            np.testing.assert_array_equal(matrix, reconstructed_matrix)


if __name__ == '__main__':
    main()
