import logging
import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from model.character_cnn_utils import CharacterIndexer
from sklearn.metrics import adjusted_rand_score

from analysis import aggregator as ag
from analysis import bert_tools as bt
from analysis import clustering as cl
from data import file_handler as fh
from data.corpus_handler import CorpusName, CorpusHandler

current_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_path)

DEFAULT_LOG_LEVEL = 'INFO'
DEFAULT_MODEL_CACHE_DIR = './model_cache'
DEFAULT_MAX_CLUSTER_DISTANCE = 0.1


def build_argument_parser() -> ArgumentParser:
    """ Builds an ArgumentParser with help messages. """
    p = ArgumentParser(description="Automatic dictionary generation.",
                       formatter_class=ArgumentDefaultsHelpFormatter)

    p.add_argument('corpus_name', type=str, default=None,
                   choices=CorpusName.get_names(),
                   help="name of the base corpus for the dictionary")
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


def main(corpus_name: CorpusName, max_dist: float, model_cache: str,
         results_path: str):
    corpus = CorpusHandler(corpus_name)
    sentences = corpus.get_sentences_as_list()

    if bt.should_tokenize(sentences):
        tokenizer = bt.get_bert_tokenizer_from_cache(model_cache)
        sentences = bt.tokenize_sentences(sentences, tokenizer)
        logging.info("Tokenized and lower cased sentences.")
    else:
        sentences = bt.lower_sentences(sentences)
        logging.info("Lower cased sentences.")

    sentences = bt.add_special_tokens_to_each(sentences)
    logging.info("Added special tokens.")
    logging.info(f"First sentence: '{sentences[0]}'.")

    abs_results_path = fh.add_and_get_abs_path(results_path)
    word_vec_file_name = fh.gen_word_vec_file_name(corpus_name)
    raw_id_map_path = fh.gen_raw_id_map_file_name(corpus_name)
    if fh.does_file_exist(abs_results_path, word_vec_file_name) \
            and fh.does_file_exist(abs_results_path, raw_id_map_path):
        word_vectors = fh.load_matrix(abs_results_path, word_vec_file_name)
        id_map = fh.load_df(abs_results_path, raw_id_map_path)
        logging.info("Loaded the word vectors and raw id_map from files.")
    else:
        indexer = CharacterIndexer()
        model = bt.get_character_bert_from_cache(model_cache)
        word_vectors, id_map = bt.embed_sentences(sentences, indexer, model)

        fh.save_matrix(abs_results_path, word_vec_file_name, word_vectors)
        fh.save_df(abs_results_path, raw_id_map_path, id_map)
        logging.info("Calculated and saved the word vectors and raw id_map.")

    logging.info(f"Shape of word vectors: {word_vectors.shape}.")
    logging.info(f"Total number of tokens: {id_map.token.count()}.")
    logging.info(f"Number of unique tokens: {id_map.token.nunique()}.")

    id_map = ag.collect_references_and_word_vectors(id_map, 'token')

    dictionary = cl.cluster_vectors_per_token(word_vectors, id_map, max_dist)
    dictionary_file_name = fh.gen_dictionary_file_name(corpus_name, max_dist)
    fh.save_df(abs_results_path, dictionary_file_name, dictionary)
    logging.info(f"Saved dictionary.")

    true_senses = cl.extract_int_senses(corpus.get_tagged_tokens())
    dict_senses = cl.extract_int_senses(cl.extract_flat_senses(dictionary))
    logging.info(f"ARI={adjusted_rand_score(true_senses, dict_senses)}")


if __name__ == '__main__':
    argument_parser = build_argument_parser()
    args = argument_parser.parse_args(sys.argv[1:])

    logging.basicConfig(level=args.log.upper(),
                        format='%(levelname)s: %(message)s')

    main(corpus_name=args.corpus_name, max_dist=args.max_dist,
         model_cache=args.model_cache, results_path=args.results_path)
