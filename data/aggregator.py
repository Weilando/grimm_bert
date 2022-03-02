import pandas as pd
import torch
from transformers import BatchEncoding


def gen_sorted_distinct_list(series: pd.Series) -> list:
    """ Transforms 'series' into a sorted, distinct list. """
    return series.sort_values().unique().tolist()


def collect_references_and_word_vectors(df: pd.DataFrame) -> pd.DataFrame:
    """ Collects sorted, distinct lists of reference-ids and word-vector-ids per
    token. """
    return df.groupby(by='token')\
        .agg({'reference-id': gen_sorted_distinct_list,
              'word-vector-id': gen_sorted_distinct_list})\
        .reset_index()


def unpack_and_strip(tensor: torch.tensor) -> torch.tensor:
    """ Removes the first dimension and omits the first and last entry. """
    assert tensor.shape[0] == 1
    assert len(tensor.shape) > 1
    return tensor[0, 1:-1]


def concat_word_vectors(word_vectors: list) -> torch.tensor:
    """ Concatenates word-vectors into a matrix. Each row is a word-vector. """
    assert all(type(v) == torch.Tensor for v in word_vectors)
    assert all(len(v.shape) > 1 for v in word_vectors)

    return torch.cat(word_vectors, dim=0)


def gen_ids_for_tokens_and_references(encoded_sentences: list) -> pd.DataFrame:
    """ Generates a DataFrame with a reference-id and word-vector-id per token.
    Uses input_ids as token. """
    assert all(type(e) == BatchEncoding for e in encoded_sentences)

    tokens = [s.input_ids.flatten() for s in encoded_sentences]
    reference_ids = [torch.full_like(t, i) for i, t in enumerate(tokens)]

    df = pd.DataFrame({'token': torch.cat(tokens, dim=0).numpy(),
                       'reference-id': torch.cat(reference_ids, dim=0).numpy()})
    df['word-vector-id'] = range(len(df.index))
    return df
