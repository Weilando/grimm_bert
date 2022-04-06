import logging
import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import analysis.aggregator as ag
import analysis.clustering as cl
import analysis.pipeline_blocks as pb
import data.file_handler as fh
from data.corpus_handler import CorpusName

current_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_path)

DEFAULT_CORPUS_CACHE_DIR = './data/corpus_cache'
DEFAULT_MODEL_CACHE_DIR = './model_cache'
DEFAULT_RESULTS_PATH = './data/results'
DEFAULT_MAX_CLUSTER_DISTANCE = 0.1
DEFAULT_LOG_LEVEL = 'INFO'


def build_argument_parser() -> ArgumentParser:
    """ Builds an ArgumentParser with help messages. """
    p = ArgumentParser(description="Automatic dictionary generation.",
                       formatter_class=ArgumentDefaultsHelpFormatter)

    p.add_argument('corpus_name', type=str, default=None,
                   choices=CorpusName.get_names(),
                   help="name of the base corpus for the dictionary")

    p.add_argument('-c', '--corpus_cache', type=str, action='store',
                   default=DEFAULT_CORPUS_CACHE_DIR,
                   help="relative path from project root to corpus files")
    p.add_argument('-m', '--model_cache', type=str, action='store',
                   default=DEFAULT_MODEL_CACHE_DIR,
                   help="relative path from project root to model files")
    p.add_argument('-r', '--results_path', type=str, action='store',
                   default=DEFAULT_RESULTS_PATH,
                   help="relative path from project root to result files")
    p.add_argument('-d', '--max_dist', type=float, action='store',
                   default=DEFAULT_MAX_CLUSTER_DISTANCE,
                   help="maximum distance for clustering")
    p.add_argument('-l', '--log', type=str, action='store',
                   default=DEFAULT_LOG_LEVEL, help="logging level")

    return p


def main(corpus_name: CorpusName, corpus_cache: str, model_cache: str,
         results_path: str, max_dist: float):
    stats = {'corpus_name': corpus_name, 'max_dist': max_dist}

    abs_results_path = fh.add_and_get_abs_path(results_path)
    word_vectors, id_map = pb.get_word_vectors(
        corpus_name, corpus_cache, model_cache, abs_results_path)

    stats.update(ag.count_total_and_unique(id_map, 'token'))
    id_map = ag.collect_references_and_word_vectors(id_map, 'token')

    dictionary = cl.cluster_vectors_per_token(word_vectors, id_map, max_dist)
    logging.info(f"Generated dictionary.")
    dictionary_file_name = fh.gen_dictionary_file_name(corpus_name, max_dist)
    fh.save_df(abs_results_path, dictionary_file_name, dictionary)

    dict_senses = ag.extract_flat_senses(dictionary)
    stats.update(ag.count_total_and_unique(dict_senses, 'sense'))
    stats.update(pb.evaluate_with_ari(corpus_name, corpus_cache, dict_senses))

    stats_file_name = fh.gen_stats_file_name(corpus_name, max_dist)
    fh.save_stats(abs_results_path, stats_file_name, stats)


if __name__ == '__main__':
    argument_parser = build_argument_parser()
    args = argument_parser.parse_args(sys.argv[1:])

    logging.basicConfig(level=args.log.upper(),
                        format='%(levelname)s: %(message)s')

    main(corpus_name=args.corpus_name, corpus_cache=args.corpus_cache,
         model_cache=args.model_cache, results_path=args.results_path,
         max_dist=args.max_dist)
