import sys
from argparse import ArgumentParser

from sklearn.metrics.pairwise import cosine_distances as pw_cos_distance
from transformers import BertModel, BertTokenizer

import analysis.bert_tools as abt
import analysis.clustering as ac
import data.aggregator as da


def log(message, verbose):
    if verbose:
        print(message)


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
    id_map = da.gen_ids_for_tokens_and_references(encoded_sentences)

    return word_vectors.numpy(), id_map


def should_print_help(args):
    return len(args) < 1


def parse_arguments(args):
    """ Parses arguments using an ArgumentParser with help messages. """
    parser = ArgumentParser(description="Automatic dictionary generation.")

    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help="activate word_vectors")
    parser.add_argument('-m', '--model_name', type=str, action='store',
                        default='bert-base-cased',
                        help="name of the applied Huggingface model")
    parser.add_argument('-p', '--p_norm', type=int, action='store', default=2,
                        help="p-norm for the similarity matrix of word-vectors")

    if should_print_help(args):
        parser.print_help(sys.stderr)
        sys.exit()
    return parser.parse_args(args)


def main(args):
    parsed_args = parse_arguments(args)
    log(parsed_args, parsed_args.verbose)
    sentences = ["He wears a watch.", "She glances at her watch.",
                 "He wants to watch the soccer match."]
    word_vectors, id_map = parse_sentences(sentences, parsed_args.model_name)
    log(f"word_vectors: {word_vectors.shape}", parsed_args.verbose)
    log(f"tokens: {id_map.token.count()}", parsed_args.verbose)

    id_map_reduced = da.collect_references_and_word_vectors(id_map, 'token')
    log(f"unique tokens: {id_map_reduced.token.count()}", parsed_args.verbose)

    distance_matrix = pw_cos_distance(word_vectors)
    log(f"distance matrix: {distance_matrix.shape}", parsed_args.verbose)

    id_map_word_vector = ac.cluster_feature('word_vector_id', distance_matrix,
                                            id_map, id_map_reduced)
    id_map_word_vector = abt.add_decoded_tokens(id_map_word_vector,
                                                parsed_args.model_name)
    log(f"By word_vector_id:\n{id_map_word_vector}", parsed_args.verbose)

    id_map_reference = ac.cluster_feature('reference_id', distance_matrix,
                                          id_map, id_map_reduced)
    id_map_reference = abt.add_decoded_tokens(id_map_reference,
                                              parsed_args.model_name)
    log(f"By reference_id:\n{id_map_reference}", parsed_args.verbose)


if __name__ == '__main__':
    main(sys.argv[1:])
