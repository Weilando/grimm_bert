from typing import Union, List, Dict

import numpy as np
import pandas as pd
import torch


def collect_references_and_word_vectors(
        df: pd.DataFrame, by: Union[str, List[str]]) -> pd.DataFrame:
    """ Collects lists of reference-ids and word-vector-ids per token. """
    return df.groupby(by=by) \
        .agg({'reference_id': list, 'word_vector_id': list}) \
        .reset_index()


def unpack_per_word_vector(df: pd.DataFrame, to_unpack: List[str]) \
        -> pd.DataFrame:
    """ Unpacks lists in the columns (e.g., 'reference_id' and 'word_vector_id')
    and sorts all rows by the word_vector_ids. """
    return df.explode(to_unpack) \
        .infer_objects() \
        .sort_values('word_vector_id') \
        .reset_index(drop=True)


def concat_word_vectors(word_vectors: List[torch.Tensor]) -> torch.tensor:
    """ Concatenates word-vectors into a matrix. Each row is a word-vector. """
    assert all(len(v.shape) > 1 for v in word_vectors)

    return torch.cat(word_vectors, dim=0)


def gen_ids_for_vectors_and_references(tokenized_sentences: List[List[str]]) \
        -> pd.DataFrame:
    """ Generates ids for references and word vectors per token. """
    tokens = np.concatenate(tokenized_sentences, axis=0)
    word_vector_ids = range(len(tokens))
    ref_ids = np.concatenate(
        [np.full_like(t, i, int) for i, t in enumerate(tokenized_sentences)],
        axis=0)

    return pd.DataFrame({'token': tokens, 'reference_id': ref_ids,
                         'word_vector_id': word_vector_ids})


def extract_flat_senses(dictionary: pd.DataFrame) -> pd.DataFrame:
    """ Extracts senses per and sorts by word_vector_id from 'dictionary'. Drops
    other columns. """
    return unpack_per_word_vector(dictionary[['word_vector_id', 'sense']],
                                  ['word_vector_id', 'sense']) \
        .set_index('word_vector_id')


def extract_int_senses(dictionary: pd.DataFrame) -> List[int]:
    """ Enumerates unique senses and returns an array of those sense ids.
    Flattens and sorts word_vector_ids and senses if they are lists. """
    return dictionary.sense.factorize()[0].tolist()


def count_total_and_unique(df: pd.DataFrame, column: str) -> Dict:
    """ Calculates the total and distinct number of elements in 'column'. """
    return {f"total_{column}_count": int(df[column].count()),
            f"unique_{column}_count": int(df[column].nunique())}


def count_unique_senses_per_token(df: pd.DataFrame) -> pd.DataFrame:
    """ Counts unique senses per token. """
    return df.groupby(by='token') \
        .agg(n_senses=('sense', 'nunique')) \
        .reset_index()


def add_sense_counts_to_id_map(id_map: pd.DataFrame,
                               sense_counts: pd.DataFrame) -> pd.DataFrame:
    """ Adds the number of senses from 'sense_counts' to 'id_map'. """
    return pd.merge(id_map, sense_counts, on='token')
