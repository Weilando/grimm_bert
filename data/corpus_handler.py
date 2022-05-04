from typing import List

from pandas import DataFrame

from data.corpus_name import CorpusName
from data.file_handler import add_and_get_abs_path, load_df
from data.file_name_generator import gen_sentences_file_name, \
    gen_tagged_tokens_file_name


class CorpusHandler(object):
    def __init__(self, corpus_name: CorpusName, corpus_cache_path: str):
        self.corpus_name = corpus_name
        self.corpus_path = add_and_get_abs_path(corpus_cache_path)
        self.sentences_name = gen_sentences_file_name(corpus_name)
        self.tagged_tokens_name = gen_tagged_tokens_file_name(corpus_name)

    def get_sentences(self) -> DataFrame:
        """ Loads tokenized sentences from file. """
        return load_df(self.corpus_path, self.sentences_name)

    def get_sentences_as_list(self) -> List[List[str]]:
        """ Returns a list of sentences. Each sentence is a list of tokens. """
        return self.get_sentences().sentence.tolist()

    def get_tagged_tokens(self) -> DataFrame:
        """ Loads tokens and corresponding senses from file. """
        return load_df(self.corpus_path, self.tagged_tokens_name)
