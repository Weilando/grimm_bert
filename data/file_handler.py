import json
import os
from typing import Dict

import numpy as np
import pandas as pd


def add_and_get_abs_path(relative_path: str = '/data') -> os.path:
    """ Generates an absolute path and add missing directories. """
    absolute_path = os.path.join(os.getcwd(), relative_path)
    if not os.path.exists(absolute_path):
        os.mkdir(absolute_path)
    return absolute_path


def does_file_exist(absolute_path: os.path, file_name: str) -> bool:
    """ Indicates if 'file_name' exists at 'absolute_path'. """
    file_path = os.path.join(absolute_path, file_name)
    return os.path.exists(file_path) and os.path.isfile(file_path)


def gen_dictionary_file_name(corpus_name: str, linkage: str, distance: float) \
        -> str:
    """ Generates the name for a file holding a dictionary DataFrame. """
    return f"{corpus_name}-linkage_{linkage}-dist_{distance}-dictionary.pkl"


def gen_raw_id_map_file_name(corpus_name: str) -> str:
    """ Generates the name for a file holding a raw id_map DataFrame. """
    return f"{corpus_name}-raw_id_map.pkl"


def gen_sentences_file_name(corpus_name: str) -> str:
    """ Generates the name for a file of a DataFrame with sentences. """
    return f"{corpus_name}-sentences.pkl"


def gen_stats_file_name(corpus_name: str, linkage: str, distance: float) -> str:
    """ Generates the name for a file holding a dict with statistics. """
    return f"{corpus_name}-linkage_{linkage}-dist_{distance}-stats.json"


def gen_tagged_tokens_file_name(corpus_name: str) -> str:
    """ Generates the name for a file of a DataFrame with tokens and tags. """
    return f"{corpus_name}-tagged_tokens.pkl"


def gen_word_vec_file_name(corpus_name: str) -> str:
    """ Generates the name for a file holding a word vector matrix. """
    return f"{corpus_name}-word_vectors.npz"


def load_df(absolute_path: os.path, file_name: str) -> pd.DataFrame:
    """ Loads a DataFrame from pkl-file 'file_name' at 'absolute_path'. """
    file_path = os.path.join(absolute_path, file_name)
    return pd.read_pickle(file_path)


def load_matrix(absolute_path: os.path, file_name: str) -> np.ndarray:
    """ Loads a matrix from npz-file 'file_name' at 'absolute_path'. """
    file_path = os.path.join(absolute_path, file_name)

    with np.load(file_path) as result_file:
        return result_file['m']


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


def save_stats(absolute_path: os.path, file_name: str, stats: Dict):
    """ Saves 'stats' in a json-file with 'file_name' at 'absolute_path'. """
    file_path = os.path.join(absolute_path, file_name)

    with open(file_path, "w") as f:
        json.dump(stats, f, indent=2)
