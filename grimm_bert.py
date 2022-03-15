import logging
import os
import sys
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from sklearn.metrics.pairwise import cosine_distances as pw_cos_distance
from transformers import BertModel, BertTokenizer

import analysis.bert_tools as abt
import analysis.clustering as ac
import data.aggregator as da
import data.result_handler as rh

current_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_path)

DEFAULT_LOG_LEVEL = 'INFO'
DEFAULT_MODEL_CACHE_PATH = './model_cache'
DEFAULT_MODEL_NAME = 'bert-base-cased'
DEFAULT_MAX_CLUSTER_DISTANCE = 0.1


def parse_sentences(sentences: list, model_name: str) -> tuple:
    """ Parses 'sentences' with a tokenizer according to 'model_name'. Returns a
    tensor with one word-vector per token in each row, and a lookup table with
    reference-ids and word-vector-ids per token. """
    assert all([type(s) == str for s in sentences])

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    encoded_sentences = [abt.encode_text(s, tokenizer) for s in sentences]
    word_vectors = [abt.calc_word_vectors(e, model).squeeze(dim=0) for e in
                    encoded_sentences]

    word_vectors = da.concat_word_vectors(word_vectors)
    id_map = da.gen_ids_for_vectors_and_references(encoded_sentences)

    return word_vectors.numpy(), id_map


def build_argument_parser() -> ArgumentParser:
    """ Builds an ArgumentParser with help messages. """
    p = ArgumentParser(description="Automatic dictionary generation.",
                       formatter_class=ArgumentDefaultsHelpFormatter)

    p.add_argument('results_path', type=str, default=None,
                   help="relative path from project root to result files")

    p.add_argument('-l', '--log', type=str, action='store',
                   default=DEFAULT_LOG_LEVEL, help="logging level")
    p.add_argument('-m', '--model_name', type=str, action='store',
                   default=DEFAULT_MODEL_NAME,
                   help="name of the Huggingface model")
    p.add_argument('-c', '--model_cache', type=str, action='store',
                   default=DEFAULT_MODEL_CACHE_PATH,
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

    if abt.should_download_model(parsed_args.model_name,
                                 parsed_args.model_cache_path):
        abt.download_and_cache_model(parsed_args.model_name,
                                     parsed_args.model_cache_path)

    sentences = ["He wears a watch.", "She glances at her watch.",
                 "He wants to watch the soccer match."]
    word_vectors, id_map = parse_sentences(sentences, parsed_args.model_name)
    logging.info(f"Shape of word-vectors is {word_vectors.shape}.")

    id_map_reduced = da.agg_references_and_word_vectors(id_map, 'token')
    logging.info(f"Number of unique tokens is {id_map_reduced.token.count()}.")

    distance_matrix = pw_cos_distance(word_vectors)

    dictionary = ac.cluster_vectors_per_token(distance_matrix, id_map,
                                              id_map_reduced,
                                              parsed_args.max_dist)
    dictionary = abt.add_decoded_tokens(dictionary, parsed_args.model_name)
    print(f"Dictionary for max_dist={parsed_args.max_dist}:\n{dictionary}")

    save_time = time.strftime("%Y_%m_%d-%H_%M_%S", time.localtime())
    rh.save_results(save_time, parsed_args.results_path, distance_matrix,
                    dictionary)
    logging.info(f"Saved results at {parsed_args.results_path}/{save_time}*.")


if __name__ == '__main__':
    main(sys.argv[1:])
