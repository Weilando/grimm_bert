from typing import Union, List, Dict

import numpy as np
import pandas as pd

from data.corpus_handler import CorpusHandler


def pack_sentence_ids_and_token_ids(
        df: pd.DataFrame, by: Union[str, List[str]]) -> pd.DataFrame:
    """ Collects lists of sentence_ids and token_ids per unique token. """
    return df.groupby(by=by) \
        .agg({'sentence_id': list, 'token_id': list}) \
        .reset_index()


def unpack_and_sort_per_token_id(
        df: pd.DataFrame, to_unpack: List[str]) -> pd.DataFrame:
    """ Unpacks lists in the columns (e.g., 'sentence_id' and 'token_id') and
    sorts all rows by the token_ids. """
    return df.explode(to_unpack) \
        .infer_objects() \
        .sort_values('token_id') \
        .reset_index(drop=True)


def gen_ids_for_sentences_and_tokens(tokenized_sentences: List[List[str]]) \
        -> pd.DataFrame:
    """ Generates ids for sentences and tokens. """
    tokens = np.concatenate(tokenized_sentences, axis=0)
    token_ids = range(len(tokens))
    sentence_ids = np.concatenate(
        [np.full_like(t, i, int) for i, t in enumerate(tokenized_sentences)],
        axis=0)

    return pd.DataFrame({'token': tokens, 'sentence_id': sentence_ids,
                         'token_id': token_ids})


def extract_flat_senses(dictionary: pd.DataFrame) -> pd.DataFrame:
    """ Extracts senses per and sorts by token_id from 'dictionary'. Drops
    other columns. """
    return unpack_and_sort_per_token_id(dictionary[['token_id', 'sense']],
                                        ['token_id', 'sense']) \
        .set_index('token_id')


def extract_int_senses_from_df(df: pd.DataFrame) -> np.ndarray:
    """ Enumerates unique senses and returns a list of those sense ids.
    Flattens and sorts token_ids and senses if they are lists. """
    return df.sense.factorize()[0]


def extract_int_senses_from_list(senses: List) -> np.ndarray:
    """ Enumerates unique senses and returns a list of those sense ids. """
    return pd.factorize(senses)[0]


def count_total_and_unique(df: pd.DataFrame, column: str) -> Dict:
    """ Calculates the total and distinct number of elements in 'column'. """
    return {f"total_{column}_count": int(df[column].count()),
            f"unique_{column}_count": int(df[column].nunique())}


def count_unique_senses_per_token(df: pd.DataFrame) -> pd.DataFrame:
    """ Counts occurrences and unique senses per token. """
    return df.groupby(by='token') \
        .agg(unique_sense_count=('sense', 'nunique'),
             total_token_count=('token', 'count')) \
        .reset_index()


def add_sense_counts_to_id_map(id_map: pd.DataFrame,
                               sense_counts: pd.DataFrame) -> pd.DataFrame:
    """ Adds the number of senses from 'sense_counts' to 'id_map'. """
    return pd.merge(id_map, sense_counts, on='token')


def count_tokens_per_sense_count(sense_counts: pd.DataFrame) -> pd.DataFrame:
    """ Aggregates total and unique token counts per sense count. """
    return sense_counts.groupby(by='unique_sense_count') \
        .agg(unique_token_count=('token', 'nunique'),
             total_token_count=('total_token_count', 'sum')) \
        .reset_index()


def count_monosemous_and_polysemous_tokens(sense_counts: pd.DataFrame) -> Dict:
    """ Counts the total and distinct number of tokens with exactly one and more
    than one unique sense. """
    poly = sense_counts[sense_counts.unique_sense_count > 1]
    mono = sense_counts[sense_counts.unique_sense_count == 1]
    return {'total_monosemous_token_count': mono.total_token_count.sum(),
            'unique_monosemous_token_count': mono.token.nunique(),
            'total_polysemous_token_count': poly.total_token_count.sum(),
            'unique_polysemous_token_count': poly.token.nunique()}


def calc_corpus_statistics_for_tagged_senses(corpus: CorpusHandler) -> Dict:
    """ Calculates statistics for tokens and senses in 'corpus'. Only considers
    tokens with given sense tags as polysemous or monosemous tokens."""
    stats = {'sentence_count': corpus.get_sentences().sentence.count()}

    tagged_tokens = corpus.get_tagged_tokens()
    stats.update(count_total_and_unique(tagged_tokens, 'token'))

    tagged_tokens = tagged_tokens[tagged_tokens.tagged_sense]
    stats.update(count_total_and_unique(tagged_tokens, 'sense'))
    tagged_stats = count_total_and_unique(tagged_tokens, 'token')
    stats.update({
        'total_tagged_token_count': tagged_stats['total_token_count'],
        'unique_tagged_token_count': tagged_stats['unique_token_count']})

    sense_counts_per_token = count_unique_senses_per_token(tagged_tokens)
    stats.update(count_monosemous_and_polysemous_tokens(sense_counts_per_token))

    return stats
