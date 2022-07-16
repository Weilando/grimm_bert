import logging

import pandas as pd

from aggregation import pipeline_blocks as pb, aggregator as ag
from clustering import hierarchical_clustering as hc
from clustering.linkage_name import LinkageName
from clustering.metric_name import MetricName
from data import file_handler as fh, file_name_generator as fg
from data.corpus_handler import CorpusHandler


def create_dictionary_with_known_sense_counts(
        corpus: CorpusHandler, model_cache: str, abs_results_path: str,
        affinity_name: MetricName, linkage_name: LinkageName,
        experiment_name: str, stats: dict) -> pd.DataFrame:
    """ Creates a dictionary from the given corpus with word vectors from
    CharacterBERT and hierarchical clustering with the specified affinity
    metric, linkage criterion and the true unique sense count per token. """
    word_vectors, id_map = pb.get_word_vectors(corpus, model_cache,
                                               abs_results_path)

    tagged_tokens = corpus.get_tagged_tokens()
    id_map = id_map[tagged_tokens.tagged_sense]
    tagged_tokens = tagged_tokens[tagged_tokens.tagged_sense]
    stats.update(ag.count_total_and_unique(id_map, 'token'))
    id_map = ag.pack_sentence_ids_and_token_ids(id_map, 'token')
    id_map = pb.add_sense_counts_to_id_map(tagged_tokens, id_map)

    dictionary = hc.cluster_vectors_per_token_with_known_sense_count(
        word_vectors, id_map, affinity_name, linkage_name)
    logging.info(f"Generated dictionary.")
    fh.save_df(abs_results_path,
               fg.gen_dictionary_file_name(experiment_name),
               dictionary)

    dict_senses = ag.extract_flat_senses(dictionary)
    stats.update(ag.count_total_and_unique(dict_senses, 'sense'))
    stats.update(pb.calc_ari(tagged_tokens, dict_senses))
    fh.save_stats(
        abs_results_path, fg.gen_stats_file_name(experiment_name), stats)

    return dictionary


def create_dictionary_with_max_distance(
        corpus: CorpusHandler, model_cache: str, abs_results_path: str,
        affinity_name: MetricName, linkage_name: LinkageName,
        max_distance: float, experiment_name: str, stats: dict) -> pd.DataFrame:
    """ Creates a dictionary from the given corpus with word vectors from
    CharacterBERT and hierarchical clustering with the specified affinity
    metric, linkage criterion and maximum distance. """
    word_vectors, id_map = pb.get_word_vectors(corpus, model_cache,
                                               abs_results_path)

    stats.update(ag.count_total_and_unique(id_map, 'token'))
    id_map = ag.pack_sentence_ids_and_token_ids(id_map, 'token')

    dictionary = hc.cluster_vectors_per_token_with_max_distance(
        word_vectors, id_map, affinity_name, linkage_name, max_distance)
    logging.info(f"Generated dictionary.")
    fh.save_df(abs_results_path,
               fg.gen_dictionary_file_name(experiment_name),
               dictionary)

    dict_senses = ag.extract_flat_senses(dictionary)
    stats.update(ag.count_total_and_unique(dict_senses, 'sense'))
    stats.update(pb.calc_ari(corpus.get_tagged_tokens(), dict_senses))
    fh.save_stats(
        abs_results_path, fg.gen_stats_file_name(experiment_name), stats)

    return dictionary


def create_dictionary_with_min_silhouette(
        corpus: CorpusHandler, model_cache: str, abs_results_path: str,
        affinity_name: MetricName, linkage_name: LinkageName,
        min_silhouette: float, experiment_name: str, stats: dict) \
        -> pd.DataFrame:
    """ Creates a dictionary from the given corpus with word vectors from
    CharacterBERT and hierarchical clustering with the specified affinity
    metric, linkage criterion. Iteratively increases the sense count per token
    and takes the clustering with the highest Silhouette Coefficient. Presumes
    'min_silhouette' for a single cluster. """
    word_vectors, id_map = pb.get_word_vectors(corpus, model_cache,
                                               abs_results_path)

    stats.update(ag.count_total_and_unique(id_map, 'token'))
    id_map = ag.pack_sentence_ids_and_token_ids(id_map, 'token')

    dictionary = hc.cluster_vectors_per_token_with_silhouette_criterion(
        word_vectors, id_map, affinity_name, linkage_name, min_silhouette)
    logging.info(f"Generated dictionary.")
    fh.save_df(abs_results_path,
               fg.gen_dictionary_file_name(experiment_name),
               dictionary)

    dict_senses = ag.extract_flat_senses(dictionary)
    stats.update(ag.count_total_and_unique(dict_senses, 'sense'))
    stats.update(pb.calc_ari(corpus.get_tagged_tokens(), dict_senses))
    fh.save_stats(
        abs_results_path, fg.gen_stats_file_name(experiment_name), stats)

    return dictionary
