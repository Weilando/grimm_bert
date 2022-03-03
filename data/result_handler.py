import os

import numpy as np
import pandas as pd


def add_and_get_abs_path(relative_path: str = '/data/results') -> os.path:
    """ Generates an absolute path and add missing directories. """
    absolute_path = os.path.join(os.getcwd(), relative_path)
    if not os.path.exists(absolute_path):
        os.mkdir(absolute_path)
    return absolute_path


def gen_df_file_name(name: str) -> str:
    """ Generates the name for a file holding a DataFrame. """
    return f"{name}-df.pkl"


def gen_matrix_file_name(name: str) -> str:
    """ Generates the name for a file holding a matrix. """
    return f"{name}-matrix.npz"


def save_df(absolute_path: os.path, file_name: str, df: pd.DataFrame):
    """ Saves 'df' in a pkl-file with 'file_name' at 'absolute_path'. """
    file_path = os.path.join(absolute_path, file_name)

    with open(file_path, "wb") as f:
        pd.to_pickle(df, f)


def save_matrix(absolute_path: os.path, file_name: str, matrix: np.ndarray):
    """ Saves 'matrix' in a npz-file with 'file_name' at 'absolute_path'. """
    file_path = os.path.join(absolute_path, file_name)

    with open(file_path, "wb") as f:
        np.savez(f, m=matrix)


def save_results(name: str, rel_path: str, matrix: np.ndarray,
                 df: pd.DataFrame) -> None:
    """ Saves 'matrix' and 'df' at 'rel_path'. File-names start with 'name'. """
    result_path = add_and_get_abs_path(rel_path)
    matrix_file_name = gen_matrix_file_name(name)
    df_file_name = gen_df_file_name(name)
    save_matrix(result_path, matrix_file_name, matrix)
    save_df(result_path, df_file_name, df)


def load_df(absolute_path: os.path, file_name: str) -> pd.DataFrame:
    """ Loads a DataFrame from pkl-file 'file_name' at 'absolute_path'. """
    file_path = os.path.join(absolute_path, file_name)
    return pd.read_pickle(file_path)


def load_matrix(absolute_path: os.path, file_name: str) -> np.ndarray:
    """ Loads a matrix from npz-file 'file_name' at 'absolute_path'. """
    file_path = os.path.join(absolute_path, file_name)

    with np.load(file_path) as result_file:
        return result_file['m']
