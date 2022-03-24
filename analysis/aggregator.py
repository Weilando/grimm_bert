from typing import Union, List

import pandas as pd
import torch
from transformers import BatchEncoding


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


def gen_ids_for_vectors_and_references(encoded_sentences: List[BatchEncoding]) \
        -> pd.DataFrame:
    """ Generates a DataFrame with a reference-id and word-vector-id per token.
    Uses input_ids as token. """
    tokens = [s.input_ids.flatten() for s in encoded_sentences]
    reference_ids = [torch.full_like(t, i) for i, t in enumerate(tokens)]

    df = pd.DataFrame({'token': torch.cat(tokens, dim=0).numpy(),
                       'reference_id': torch.cat(reference_ids, dim=0).numpy()})
    df['word_vector_id'] = range(len(df.index))
    return df
