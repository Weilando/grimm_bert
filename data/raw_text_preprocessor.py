from typing import List

import pandas as pd
from transformers import BertTokenizer

from data.corpus_handler import CorpusName
from data.corpus_preprocessor import CorpusPreprocessor, STD_SENSE
from data.toy_preprocessor import flatten_list
from grimm_bert import DEFAULT_CORPUS_CACHE_DIR


def read_and_strip_lines(file_name: str) -> List[str]:
    """ Reads non-empty lines and removes trailing and leading white spaces. """
    with open(file_name) as file:
        return [line for line in (raw_line.strip() for raw_line in file)
                if line]


def tokenize_lines(lines: List[str], tokenizer: BertTokenizer) \
        -> List[List[str]]:
    """ Tokenizes each line. """
    return [tokenizer.basic_tokenizer.tokenize(line) for line in lines]


class RawTextPreprocessor(CorpusPreprocessor):
    def __init__(self, lines: List[str], tokenizer: BertTokenizer,
                 corpus_name: CorpusName = CorpusName.SHAKESPEARE,
                 corpus_cache_path: str = DEFAULT_CORPUS_CACHE_DIR):
        """ Preprocessor for corpora in a raw text format. Handles each line as
        a sentence. Strips and lowers each sentence and applies 'tokenizer'.
        Generates generic semantic tags. """
        super().__init__(corpus_name, corpus_cache_path)
        self.lines = tokenize_lines(lines, tokenizer)

    def get_sentences(self) -> pd.DataFrame:
        sentences = [[token.lower() for token in sentence]
                     for sentence in self.lines]
        return pd.DataFrame({'sentence': sentences})

    def get_tagged_tokens(self) -> pd.DataFrame:
        tokens = [token.lower() for token in flatten_list(self.lines)]
        senses = [token + STD_SENSE for token in tokens]
        return pd.DataFrame({'token': tokens, 'sense': senses,
                             'tagged_sense': False})


if __name__ == '__main__':
    raw_text_preprocessor = RawTextPreprocessor(
        read_and_strip_lines(f'data/raw_text_corpora/shakespeare.txt'),
        BertTokenizer.from_pretrained('./model_cache/bert-base-uncased/'),
        CorpusName.SHAKESPEARE)
    raw_text_preprocessor.cache_dataset()
