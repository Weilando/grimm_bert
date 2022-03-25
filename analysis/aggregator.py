from typing import Union, List

import numpy as np
import pandas as pd
import torch


def gen_sorted_distinct_list(series: pd.Series) -> List:
    """ Transforms 'series' into a sorted, distinct list. """
    return series.sort_values().unique().tolist()


def agg_references_and_word_vectors(
        df: pd.DataFrame, by: Union[str, List[str]]) -> pd.DataFrame:
    """ Collects sorted, distinct lists of reference-ids and word-vector-ids per
    token. """
    return df.groupby(by=by) \
        .agg({'reference_id': gen_sorted_distinct_list,
              'word_vector_id': gen_sorted_distinct_list}) \
        .reset_index()


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
