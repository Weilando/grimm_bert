import logging
import os
import sys
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from model.character_bert import CharacterBertModel
from model.character_cnn_utils import CharacterIndexer
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_distances
from transformers import BertTokenizer

from analysis import aggregator as ag
from analysis import bert_tools as bt
from analysis import clustering as cl
from data import result_handler as rh
from data.dataset_handler import TOY_SENSES, TOY_SENTENCES

current_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_path)

DEFAULT_LOG_LEVEL = 'INFO'
DEFAULT_MODEL_CACHE_DIR = './model_cache'
DEFAULT_MAX_CLUSTER_DISTANCE = 0.1


def build_argument_parser() -> ArgumentParser:
    """ Builds an ArgumentParser with help messages. """
    p = ArgumentParser(description="Automatic dictionary generation.",
                       formatter_class=ArgumentDefaultsHelpFormatter)

    p.add_argument('results_path', type=str, default=None,
                   help="relative path from project root to result files")

    p.add_argument('-l', '--log', type=str, action='store',
                   default=DEFAULT_LOG_LEVEL, help="logging level")
    p.add_argument('-c', '--model_cache', type=str, action='store',
                   default=DEFAULT_MODEL_CACHE_DIR,
                   help="relative path from project root to model files")
    p.add_argument('-d', '--max_dist', type=float, action='store',
                   default=DEFAULT_MAX_CLUSTER_DISTANCE,
                   help="maximum distance for clustering")

    return p


def main(args):
    argument_parser = build_argument_parser()
    parsed_args = argument_parser.parse_args(args)
    logging.basicConfig(level=parsed_args.log.upper(),
                        format='%(levelname)s: %(message)s')

    tokenizer_path = bt.gen_model_cache_location(parsed_args.model_cache,
                                                 'bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    indexer = CharacterIndexer()
    model_path = bt.gen_model_cache_location(parsed_args.model_cache,
                                             'general_character_bert')
    model = CharacterBertModel.from_pretrained(model_path)

    word_vectors, id_map = bt.parse_sentences(TOY_SENTENCES, tokenizer, indexer,
                                              model)
    logging.info(f"Shape of word-vectors is {word_vectors.shape}.")

    id_map_reduced = ag.agg_references_and_word_vectors(id_map, 'token')
    logging.info(f"Number of unique tokens is {id_map_reduced.token.count()}.")

    distance_matrix = cosine_distances(word_vectors)

    dictionary = cl.cluster_vectors_per_token(distance_matrix, id_map,
                                              id_map_reduced,
                                              parsed_args.max_dist)
    reduced_dictionary = cl.reduce_dictionary(dictionary)
    print(f"Reduced dictionary:\n{reduced_dictionary}")

    save_time = time.strftime("%Y_%m_%d-%H_%M_%S", time.localtime())
    rh.save_results(save_time, parsed_args.results_path,
                    distance_matrix, dictionary)
    logging.info(f"Saved results at {parsed_args.results_path}/{save_time}*.")

    int_senses = cl.extract_int_senses(dictionary)
    logging.info(f"ARI={adjusted_rand_score(TOY_SENSES, int_senses)}")


if __name__ == '__main__':
    main(sys.argv[1:])
