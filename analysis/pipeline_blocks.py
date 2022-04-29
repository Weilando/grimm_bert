import logging
import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from model.character_cnn_utils import CharacterIndexer
from sklearn.metrics import adjusted_rand_score

import analysis.aggregator as ag
import analysis.bert_tools as bt
import data.file_handler as fh
from data.corpus_handler import CorpusHandler


def load_and_preprocess_sentences(corpus: CorpusHandler) -> List[List[str]]:
    """ Loads raw sentences from the specified corpus, lower cases all tokens
    and adds special tokens to each sentence. """
    sentences = corpus.get_sentences_as_list()
    sentences = bt.lower_sentences(sentences)
    sentences = bt.add_special_tokens_to_each(sentences)

    logging.info("Lower cased sentences and added special tokens.")
    return sentences


def add_sense_counts_to_id_map(corpus: CorpusHandler, id_map: pd.DataFrame) \
        -> pd.DataFrame:
    """ Loads and lower cases tagged tokens from the specified corpus and adds
    the number of unique senses per token to 'id_map'. """
    tagged_tokens = corpus.get_tagged_tokens()
    tagged_tokens['token'] = tagged_tokens.token.str.lower()
    sense_counts = ag.count_unique_senses_per_token(tagged_tokens)

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
    model = bt.get_character_bert_from_cache(model_cache)
    sentences = load_and_preprocess_sentences(corpus)
    word_vectors, raw_id_map = bt.embed_sentences(sentences, indexer, model)

    return word_vectors, raw_id_map


def get_word_vectors(corpus: CorpusHandler, model_cache: str,
                     abs_results_path: os.path) \
        -> Tuple[np.ndarray, pd.DataFrame]:
    """ Loads the word vectors and corresponding raw id_map from an existing
    result file or calculates them from scratch and creates a cache. """
    word_vec_file_name = fh.gen_word_vec_file_name(corpus.corpus_name)
    raw_id_map_file_name = fh.gen_raw_id_map_file_name(corpus.corpus_name)

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


def calc_ari_for_all_senses(corpus: CorpusHandler,
                            flat_dict_senses: pd.DataFrame) -> Dict:
    """ Calculates the Adjusted Rand Index (ARI) for 'flat_dict_senses' and the
    ground truth for the given corpus and writes it into a statistics dict.
    Considers generated senses for tokens without any sense annotations. """
    true_senses = ag.extract_int_senses_from_df(corpus.get_tagged_tokens())
    dict_senses = ag.extract_int_senses_from_df(flat_dict_senses)
    ari = adjusted_rand_score(true_senses, dict_senses)

    logging.info(f"ARI (all senses): {ari}")
    return {'ari_all': ari}


def calc_ari_for_tagged_senses(corpus: CorpusHandler,
                               flat_dict_senses: pd.DataFrame) -> Dict:
    """ Calculates the Adjusted Rand Index (ARI) for 'flat_dict_senses' and the
    ground truth for the given corpus and writes it into a statistics dict.
    Only considers tokens with existing sense annotations. """
    tagged_tokens = corpus.get_tagged_tokens()
    true_senses = ag.extract_int_senses_from_df(tagged_tokens)
    dict_senses = ag.extract_int_senses_from_df(flat_dict_senses)
    ari = adjusted_rand_score(true_senses[tagged_tokens.tagged_sense.tolist()],
                              dict_senses[tagged_tokens.tagged_sense.tolist()])

    logging.info(f"ARI (tagged senses): {ari}")
    return {'ari_tagged': ari}


def calc_ari_per_token(corpus: CorpusHandler, dictionary: pd.DataFrame) \
        -> pd.DataFrame:
    """ Adds a column with an Adjusted Rand Index (ARI) per token and senses to
    'dictionary' based on the ground truth for the given corpus. Another column
    indicates if all senses for one token are tagged. """
    tagged_tokens = corpus.get_tagged_tokens()
    true_senses = np.array(ag.extract_int_senses_from_df(tagged_tokens))

    dictionary['ari'] = dictionary.apply(
        lambda r: adjusted_rand_score(true_senses[r.word_vector_id],
                                      ag.extract_int_senses_from_list(r.sense)),
        axis=1)
    dictionary['tagged_token'] = dictionary.apply(
        lambda r: all(tagged_tokens.tagged_sense[r.word_vector_id]),
        axis=1)

    return dictionary
