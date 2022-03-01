import sys
from argparse import ArgumentParser

from transformers import BertModel, BertTokenizer

from analysis.bert_tools import calc_word_vectors, encode_text
from data.reference_aggregator import concat_word_vectors, \
    gen_ids_for_tokens_and_references


def log(message, verbose):
    if verbose:
        print(message)


def parse_sentences(sentences: list, model_name: str) -> tuple:
    assert all([type(s) == str for s in sentences])

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    encoded_sentences = [encode_text(s, tokenizer) for s in sentences]
    word_vectors = [calc_word_vectors(e, model).squeeze(dim=0) for e in
                    encoded_sentences]

    word_vectors = concat_word_vectors(word_vectors)
    lookup_table = gen_ids_for_tokens_and_references(encoded_sentences)

    return word_vectors, lookup_table


def should_print_help(args):
    return len(args) < 1


def parse_arguments(args):
    """ Creates an ArgumentParser with help messages. """
    parser = ArgumentParser(description="Automatic dictionary generation.")

    parser.add_argument('sentence', type=str, help='input sentence')
    parser.add_argument('experiment_name', type=str, help='experiment name')
    parser.add_argument('rel_path', type=str, help="relative path for results")
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help="activate word_vectors")
    parser.add_argument('-m', '--model_name', type=str, action='store',
                        default='bert-base-cased',
                        help="name of the applied Huggingface model")

    if should_print_help(args):
        parser.print_help(sys.stderr)
        sys.exit()
    return parser.parse_args(args)


def main(args):
    parsed_args = parse_arguments(args)
    log(parsed_args, parsed_args.verbose)
    word_vectors, lookup_table = parse_sentences([parsed_args.sentence],
                                                 parsed_args.model_name)
    log(word_vectors, parsed_args.verbose)
    log(lookup_table, parsed_args.verbose)


if __name__ == '__main__':
    main(sys.argv[1:])
