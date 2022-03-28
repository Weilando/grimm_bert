from enum import Enum
from typing import List, Union

import pandas as pd

import data.file_handler as fh
import data.semcor_preprocessor as sp
import data.toy_preprocessor as tp

CORPUS_PATH = './data/corpora'


class CorpusName(str, Enum):
    TOY = 'Toy'
    SEMCOR = 'SemCor'

    @classmethod
    def get_names(cls) -> List[str]:
        return [name.value for name in cls]


class CorpusHandler(object):
    def __init__(self, corpus_name: CorpusName, corpus_path: str = CORPUS_PATH):
        self.corpus_name = corpus_name
        self.corpus_path = fh.add_and_get_abs_path(corpus_path)
        if self.corpus_name == CorpusName.TOY:
            self.sentences_name = fh.gen_df_file_name(tp.SENTENCES_NAME)
            self.tagged_tokens_name = fh.gen_df_file_name(tp.TAGGED_TOKENS_NAME)
        elif self.corpus_name == CorpusName.SEMCOR:
            self.sentences_name = fh.gen_df_file_name(sp.SENTENCES_NAME)
            self.tagged_tokens_name = fh.gen_df_file_name(sp.TAGGED_TOKENS_NAME)

    def get_sentences(self) -> pd.DataFrame:
        return fh.load_df(self.corpus_path, self.sentences_name)

    def get_sentences_as_list(self) -> Union[List[str], List[List[str]]]:
        return self.get_sentences().sentence.tolist()

    def get_tagged_tokens(self) -> pd.DataFrame:
        return fh.load_df(self.corpus_path, self.tagged_tokens_name)


def cache_corpora(corpus_path: str = CORPUS_PATH):
    """ Caches all corpora at 'corpus_path' """
    absolute_corpus_path = fh.add_and_get_abs_path(corpus_path)
    tp.cache_toy_dataset(absolute_corpus_path)
    sp.cache_semcor_dataset(absolute_corpus_path)
