import logging
import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import Union

from aggregation.pipelines import create_dictionary_with_max_distance, \
    create_dictionary_with_known_sense_counts, \
    create_dictionary_with_min_silhouette
from clustering.linkage_name import LinkageName
from clustering.metric_name import MetricName
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

    p.add_argument('experiment_name', type=str, default=None,
                   help="experiment name, used as prefix for result files")
    p.add_argument('corpus_name', type=str, default=None,
                   choices=CorpusName.get_values(),
                   help="name of the base corpus for the dictionary")
    p.add_argument('affinity_name', type=str, default=None,
                   choices=MetricName.get_values(),
                   help="name of the linkage criterion for clustering")
    p.add_argument('linkage_name', type=str, default=None,
                   choices=LinkageName.get_values(),
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
    p.add_argument('-d', '--max_distance', type=float, action='store',
                   default=None, help="maximum distance for clustering")
    p.add_argument('-s', '--min_silhouette', type=float, action='store',
                   default=None, help="presumed Silhouette Coefficient for k=1")
    p.add_argument('-k', '--known_senses', type=bool, action='store',
                   default=False, help="usage of ground truth sense counts")
    p.add_argument('-l', '--log', type=str, action='store',
                   default=DEFAULT_LOG_LEVEL, help="logging level")

    return p


def is_max_dist_defined(max_dist: Union[float, None]) -> bool:
    """ Indicates if 'max_distance' is a float greater than zero. """
    return max_dist is not None and max_dist > 0.0


def is_min_silhouette_defined(min_silhouette: Union[float, None]) -> bool:
    """ Indicates if 'min_silhouette' is a float between zero and one. """
    return min_silhouette is not None and 0.0 <= min_silhouette <= 1.0


if __name__ == '__main__':
    argument_parser = build_argument_parser()
    args = argument_parser.parse_args(sys.argv[1:])

    logging.basicConfig(level=args.log.upper(),
                        format='%(levelname)s: %(message)s')
    corpus_handler = CorpusHandler(args.corpus_name, args.corpus_cache)

    if is_max_dist_defined(args.max_distance):
        create_dictionary_with_max_distance(
            corpus=corpus_handler, model_cache=args.model_cache,
            results_path=args.results_path, affinity_name=args.affinity_name,
            linkage_name=args.linkage_name, max_distance=args.max_distance,
            experiment_name=args.experiment_name)
    if is_min_silhouette_defined(args.min_silhouette):
        create_dictionary_with_min_silhouette(
            corpus=corpus_handler, model_cache=args.model_cache,
            results_path=args.results_path, affinity_name=args.affinity_name,
            linkage_name=args.linkage_name, min_silhouette=args.min_silhouette,
            experiment_name=args.experiment_name)
    if args.known_senses:
        create_dictionary_with_known_sense_counts(
            corpus=corpus_handler, model_cache=args.model_cache,
            results_path=args.results_path, affinity_name=args.affinity_name,
            linkage_name=args.linkage_name,
            experiment_name=args.experiment_name)
