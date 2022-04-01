from typing import List

import pandas as pd

import data.file_handler as fh
from data.corpus_name import CorpusName


class CorpusHandler(object):
    def __init__(self, corpus_name: CorpusName, corpus_cache_path: str):
        self.corpus_name = corpus_name
        self.corpus_path = fh.add_and_get_abs_path(corpus_cache_path)
        self.sentences_name = fh.gen_sentences_file_name(corpus_name)
        self.tagged_tokens_name = fh.gen_tagged_tokens_file_name(corpus_name)

    def get_sentences(self) -> pd.DataFrame:
        """ Loads tokenized sentences from file. """
        return fh.load_df(self.corpus_path, self.sentences_name)

    def get_sentences_as_list(self) -> List[List[str]]:
        """ Returns a list of sentences. Each sentence is a list of tokens. """
        return self.get_sentences().sentence.tolist()

    def get_tagged_tokens(self) -> pd.DataFrame:
        """ Loads tokens and corresponding senses from file. """
        return fh.load_df(self.corpus_path, self.tagged_tokens_name)
