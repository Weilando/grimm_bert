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
from data.corpus_name import CorpusName


def load_and_preprocess_sentences(corpus_name: CorpusName, corpus_cache: str) \
        -> List[List[str]]:
    """ Loads raw sentences from the specified corpus, lowercase all tokens and
    add special tokens to each sentence. """
    corpus = CorpusHandler(corpus_name, corpus_cache)
    sentences = corpus.get_sentences_as_list()
    sentences = bt.lower_sentences(sentences)
    sentences = bt.add_special_tokens_to_each(sentences)

    logging.info("Lower cased sentences and added special tokens.")
    return sentences


def add_sense_counts_to_id_map(corpus_name: CorpusName, corpus_cache: str,
                               id_map: pd.DataFrame) -> pd.DataFrame:
    """ Loads tagged tokens from the specified corpus and adds the number of
    unique senses per token to 'id_map'. """
    corpus = CorpusHandler(corpus_name, corpus_cache)
    tagged_tokens = corpus.get_tagged_tokens()
    sense_counts = ag.count_unique_senses_per_token(tagged_tokens)

    logging.info("Loaded ground truth number of senses per token.")
    return ag.add_sense_counts_to_id_map(id_map, sense_counts)


def does_word_vector_cache_exist(abs_results_path: os.path,
                                 word_vec_file_name: str,
                                 raw_id_map_path: str) -> bool:
    """ Indicates if word vectors and id_map can be loaded from cache. """
    return (fh.does_file_exist(abs_results_path, word_vec_file_name)
            and fh.does_file_exist(abs_results_path, raw_id_map_path))


def calculate_word_vectors(corpus_name: CorpusName, corpus_cache: str,
                           model_cache: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """ Calculates word vectors with CharacterBERT and generates an id_map. """
    indexer = CharacterIndexer()
    model = bt.get_character_bert_from_cache(model_cache)
    sentences = load_and_preprocess_sentences(corpus_name, corpus_cache)
    word_vectors, raw_id_map = bt.embed_sentences(sentences, indexer, model)

    return word_vectors, raw_id_map


def get_word_vectors(corpus_name: CorpusName, corpus_cache: str,
                     model_cache: str, abs_results_path: os.path) \
        -> Tuple[np.ndarray, pd.DataFrame]:
    """ Loads the word vectors and corresponding raw id_map from an existing
    result file or calculates them from scratch and creates a cache. """
    word_vec_file_name = fh.gen_word_vec_file_name(corpus_name)
    raw_id_map_file_name = fh.gen_raw_id_map_file_name(corpus_name)

    if does_word_vector_cache_exist(abs_results_path, word_vec_file_name,
                                    raw_id_map_file_name):
        word_vectors = fh.load_matrix(abs_results_path, word_vec_file_name)
        id_map = fh.load_df(abs_results_path, raw_id_map_file_name)
        logging.info("Loaded the word vectors and raw id_map from files.")
    else:
        word_vectors, id_map = calculate_word_vectors(
            corpus_name, corpus_cache, model_cache)
        fh.save_matrix(abs_results_path, word_vec_file_name, word_vectors)
        fh.save_df(abs_results_path, raw_id_map_file_name, id_map)
        logging.info("Calculated and cached the word vectors and raw id_map.")

    return word_vectors, id_map


def evaluate_with_ari(corpus_name: CorpusName, corpus_cache: str,
                      flat_dict_senses: pd.DataFrame) -> Dict:
    """ Calculates the Adjusted Rand Index (ARI) for 'flat_dict_senses' and the
    ground truth for the given corpus and writes into a statistics dict. """
    corpus = CorpusHandler(corpus_name, corpus_cache)
    true_senses = ag.extract_int_senses(corpus.get_tagged_tokens())
    dict_senses = ag.extract_int_senses(flat_dict_senses)
    ari = adjusted_rand_score(true_senses, dict_senses)

    logging.info(f"ARI: {ari}")
    return {'ari': ari}
