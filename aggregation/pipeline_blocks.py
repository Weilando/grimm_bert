import logging
import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from model.character_bert.character_cnn_utils import CharacterIndexer
from sklearn.metrics import adjusted_rand_score, silhouette_score

import aggregation.aggregator as ag
import data.file_handler as fh
import data.file_name_generator as fg
import model.model_tools as mt
from clustering.metric_name import MetricName
from data.corpus_handler import CorpusHandler


def load_and_preprocess_sentences(corpus: CorpusHandler) -> List[List[str]]:
    """ Loads raw sentences from the specified corpus, lower cases all tokens
    and adds special tokens to each sentence. """
    sentences = corpus.get_sentences_as_list()
    sentences = mt.lower_sentences(sentences)
    sentences = mt.add_special_tokens_to_each(sentences)

    logging.info("Lower cased sentences and added special tokens.")
    return sentences


def add_sense_counts_to_id_map(tagged_tokens: pd.DataFrame,
                               id_map: pd.DataFrame) -> pd.DataFrame:
    """ Loads and lower cases tagged tokens from the specified corpus and adds
    the number of unique senses per token to 'id_map'. """
    sense_counts = ag.count_unique_senses_per_token(
        tagged_tokens[tagged_tokens.tagged_sense])

    logging.info("Loaded ground truth number of senses per token.")
    return ag.add_sense_counts_to_id_map(id_map, sense_counts)


def does_word_vector_cache_exist(abs_results_path: os.path,
                                 word_vec_file_name: str,
                                 raw_id_map_path: str) -> bool:
    """ Indicates if word vectors and id_map can be loaded from cache. """
    return (fh.does_file_exist(abs_results_path, word_vec_file_name)
            and fh.does_file_exist(abs_results_path, raw_id_map_path))


def calculate_word_vectors(corpus: CorpusHandler, model_cache: str) \
        -> Tuple[np.ndarray, pd.DataFrame]:
    """ Calculates word vectors with CharacterBERT and generates an id_map. """
    indexer = CharacterIndexer()
    model = mt.get_character_bert_from_cache(model_cache)
    sentences = load_and_preprocess_sentences(corpus)
    word_vectors, raw_id_map = mt.embed_sentences(sentences, indexer, model)

    return word_vectors, raw_id_map


def get_word_vectors(corpus: CorpusHandler, model_cache: str,
                     abs_results_path: os.path) \
        -> Tuple[np.ndarray, pd.DataFrame]:
    """ Loads the word vectors and corresponding raw id_map from an existing
    result file or calculates them from scratch and creates a cache. """
    word_vec_file_name = fg.gen_word_vec_file_name(corpus.corpus_name)
    raw_id_map_file_name = fg.gen_raw_id_map_file_name(corpus.corpus_name)

    if does_word_vector_cache_exist(abs_results_path, word_vec_file_name,
                                    raw_id_map_file_name):
        word_vectors = fh.load_matrix(abs_results_path, word_vec_file_name)
        id_map = fh.load_df(abs_results_path, raw_id_map_file_name)
        logging.info("Loaded the word vectors and raw id_map from files.")
    else:
        word_vectors, id_map = calculate_word_vectors(corpus, model_cache)
        fh.save_matrix(abs_results_path, word_vec_file_name, word_vectors)
        fh.save_df(abs_results_path, raw_id_map_file_name, id_map)
        logging.info("Calculated and cached the word vectors and raw id_map.")

    return word_vectors, id_map


def calc_ari(tagged_tokens: pd.DataFrame,
             flat_dict_senses: pd.DataFrame) -> Dict:
    """ Calculates the Adjusted Rand Index (ARI) for 'flat_dict_senses' and the
    ground truth for the given corpus and writes it into a statistics dict.
    Only considers tokens with existing sense annotations. """
    tag_mask = tagged_tokens.tagged_sense.tolist()
    true_senses = ag.extract_int_senses_from_df(tagged_tokens[tag_mask])
    dict_senses = ag.extract_int_senses_from_df(flat_dict_senses[tag_mask])
    ari = adjusted_rand_score(true_senses, dict_senses)

    logging.info(f"ARI: {ari}")
    return {'ari': ari}


def calc_ari_per_token(tagged_tokens: pd.DataFrame, dictionary: pd.DataFrame) \
        -> pd.DataFrame:
    """ Adds a column with an Adjusted Rand Index (ARI) per token and senses to
    'dictionary' based on the ground truth from 'tagged_tokens'. Another column
    indicates if all senses for one token are tagged. """
    true_senses = np.array(ag.extract_int_senses_from_df(tagged_tokens))

    dictionary['ari'] = dictionary.apply(
        lambda r: adjusted_rand_score(true_senses[r.token_id],
                                      ag.extract_int_senses_from_list(r.sense)),
        axis=1)
    dictionary['tagged_token'] = dictionary.apply(
        lambda r: all(tagged_tokens.tagged_sense[r.token_id]),
        axis=1)

    return dictionary


def calc_silhouette_score_per_sample(word_vectors: np.array, labels: np.ndarray,
                                     metric: MetricName) -> float:
    """ Calculates the Silhouette Coefficient for the given clustering or NaN,
    if the number of unique labels is invalid. The score is defined for
    2 <= n_labels <= n_samples - 1. """
    try:
        return silhouette_score(X=word_vectors, labels=labels, metric=metric)
    except ValueError:
        return np.NaN


def calc_silhouette_score_per_token(
        word_vectors: np.ndarray, dictionary: pd.DataFrame,
        metric: MetricName) -> pd.DataFrame:
    """ Adds a column with a Silhouette Coefficient per token and senses to
    'dictionary' based on the given 'metric'. Does not treat generated tokens
    differently. """
    dictionary['silhouette_score'] = dictionary.apply(
        lambda r: calc_silhouette_score_per_sample(
            word_vectors[r.token_id],
            ag.extract_int_senses_from_list(r.sense),
            metric=metric.lower()),
        axis=1)

    return dictionary
