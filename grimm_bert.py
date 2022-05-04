import logging
import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import Union

import aggregation.aggregator as ag
import aggregation.pipeline_blocks as pb
import clustering.hierarchical_clustering as hc
import data.file_handler as fh
import data.file_name_generator as fg
from clustering.affinity_name import AffinityName
from clustering.linkage_name import LinkageName
from data.corpus_handler import CorpusName, CorpusHandler

current_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_path)

DEFAULT_CORPUS_CACHE_DIR = './data/corpus_cache'
DEFAULT_MODEL_CACHE_DIR = './model_cache'
DEFAULT_RESULTS_PATH = './data/results'
DEFAULT_LOG_LEVEL = 'INFO'


def build_argument_parser() -> ArgumentParser:
    """ Builds an ArgumentParser with help messages. """
    p = ArgumentParser(description="Automatic dictionary generation."
                                   "Uses the true number of senses per token if"
                                   "no maximum distance is given.",
                       formatter_class=ArgumentDefaultsHelpFormatter)

    p.add_argument('corpus_name', type=str, default=None,
                   choices=CorpusName.get_names(),
                   help="name of the base corpus for the dictionary")
    p.add_argument('affinity_name', type=str, default=None,
                   choices=AffinityName.get_names(),
                   help="name of the linkage criterion for clustering")
    p.add_argument('linkage_name', type=str, default=None,
                   choices=LinkageName.get_names(),
                   help="name of the linkage criterion for clustering")

    p.add_argument('-c', '--corpus_cache', type=str, action='store',
                   default=DEFAULT_CORPUS_CACHE_DIR,
                   help="relative path from project root to corpus files")
    p.add_argument('-m', '--model_cache', type=str, action='store',
                   default=DEFAULT_MODEL_CACHE_DIR,
                   help="relative path from project root to model files")
    p.add_argument('-r', '--results_path', type=str, action='store',
                   default=DEFAULT_RESULTS_PATH,
                   help="relative path from project root to result files")
    p.add_argument('-d', '--max_dist', type=float, action='store', default=None,
                   help="maximum distance for clustering")
    p.add_argument('-l', '--log', type=str, action='store',
                   default=DEFAULT_LOG_LEVEL, help="logging level")

    return p


def is_max_dist_defined(max_dist: Union[float, None]) -> bool:
    """ Indicates if 'max_dist' is a float greater than zero. """
    return max_dist is not None and max_dist > 0.0


def create_dictionary(
        corpus: CorpusHandler, model_cache: str, results_path: str,
        affinity_name: AffinityName, linkage_name: LinkageName,
        max_dist: float) -> None:
    """ Creates a dictionary from the given corpus with word vectors from
    CharacterBERT and hierarchical clustering with the specified affinity
    metric, linkage criterion and maximum distance. """
    stats = {'corpus_name': corpus.corpus_name, 'affinity_name': affinity_name,
             'linkage_name': linkage_name, 'max_dist': max_dist}

    abs_results_path = fh.add_and_get_abs_path(results_path)
    word_vectors, id_map = pb.get_word_vectors(corpus, model_cache,
                                               abs_results_path)

    stats.update(ag.count_total_and_unique(id_map, 'token'))
    id_map = ag.pack_sentence_ids_and_token_ids(id_map, 'token')

    dictionary = hc.cluster_vectors_per_token(
        word_vectors, id_map, affinity_name, linkage_name, max_dist)
    logging.info(f"Generated dictionary.")
    experiment_prefix = fg.gen_experiment_prefix(corpus.corpus_name,
                                                 affinity_name,
                                                 linkage_name, max_dist)
    fh.save_df(abs_results_path,
               fg.gen_dictionary_file_name(experiment_prefix),
               dictionary)

    dict_senses = ag.extract_flat_senses(dictionary)
    stats.update(ag.count_total_and_unique(dict_senses, 'sense'))
    stats.update(pb.calc_ari(corpus.get_tagged_tokens(), dict_senses))
    fh.save_stats(abs_results_path,
                  fg.gen_stats_file_name(experiment_prefix),
                  stats)


def create_dictionary_with_known_sense_counts(
        corpus: CorpusHandler, model_cache: str, results_path: str,
        affinity_name: AffinityName, linkage_name: LinkageName) -> None:
    """ Creates a dictionary from the given corpus with word vectors from
    CharacterBERT and hierarchical clustering with the specified affinity
    metric, linkage criterion and the true unique sense count per token. """
    stats = {'corpus_name': corpus.corpus_name, 'affinity_name': affinity_name,
             'linkage_name': linkage_name}

    abs_results_path = fh.add_and_get_abs_path(results_path)
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
    experiment_prefix = fg.gen_experiment_prefix_no_dist(corpus.corpus_name,
                                                         affinity_name,
                                                         linkage_name)
    fh.save_df(abs_results_path,
               fg.gen_dictionary_file_name(experiment_prefix),
               dictionary)

    dict_senses = ag.extract_flat_senses(dictionary)
    stats.update(ag.count_total_and_unique(dict_senses, 'sense'))
    stats.update(pb.calc_ari(tagged_tokens, dict_senses))
    fh.save_stats(abs_results_path,
                  fg.gen_stats_file_name(experiment_prefix),
                  stats)


if __name__ == '__main__':
    argument_parser = build_argument_parser()
    args = argument_parser.parse_args(sys.argv[1:])

    logging.basicConfig(level=args.log.upper(),
                        format='%(levelname)s: %(message)s')
    corpus_handler = CorpusHandler(args.corpus_name, args.corpus_cache)

    if is_max_dist_defined(args.max_dist):
        create_dictionary(
            corpus=corpus_handler, model_cache=args.model_cache,
            results_path=args.results_path, affinity_name=args.affinity_name,
            linkage_name=args.linkage_name, max_dist=args.max_dist)
    else:
        create_dictionary_with_known_sense_counts(
            corpus=corpus_handler, model_cache=args.model_cache,
            results_path=args.results_path, affinity_name=args.affinity_name,
            linkage_name=args.linkage_name)
