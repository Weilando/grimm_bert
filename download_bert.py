import logging
import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace

from transformers import BertTokenizer, BertModel

current_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_path)


def download_and_cache_model(model_name: str, cache_path: str) -> None:
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    tokenizer.save_pretrained(cache_path)
    model.save_pretrained(cache_path)


def parse_arguments(args) -> Namespace:
    """ Parses arguments using an ArgumentParser with help messages. """
    parser = ArgumentParser(description="Download and cache HuggingFace model.",
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('-p', '--cache_path', type=str,
                        default='./models/bert-base-cased',
                        help="relative path from project-root to model-files")
    parser.add_argument('-l', '--log', type=str, action='store', default='INFO',
                        help="log level like INFO or WARNING")
    parser.add_argument('-m', '--model_name', type=str, action='store',
                        default='bert-base-cased',
                        help="name of Huggingface model to download")

    return parser.parse_args(args)


def main(args):
    parsed_args = parse_arguments(args)
    logging.basicConfig(level=parsed_args.log.upper(),
                        format='%(levelname)s: %(message)s')
    download_and_cache_model(parsed_args.model_name, parsed_args.cache_path)
    logging.info(f"Cached model and tokenizer for {parsed_args.model_name}"
                 f" at {parsed_args.cache_path}.")


if __name__ == '__main__':
    main(sys.argv[1:])
