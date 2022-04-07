import pandas as pd

from data import file_handler as fh
from data.corpus_handler import CorpusName


class CorpusPreprocessor(object):
    def __init__(self, corpus_name: CorpusName, corpus_cache_path: str):
        self.corpus_name = corpus_name
        self.corpus_cache_path = fh.add_and_get_abs_path(corpus_cache_path)

    def get_sentences(self) -> pd.DataFrame:
        """ Returns a DataFrame with a column 'sentence', where each sentence is
        a list of tokens. """
        pass

    def get_tagged_tokens(self) -> pd.DataFrame:
        """ Returns a DataFrame with the columns 'token' and 'sense'. """
        pass

    def cache_dataset(self) -> None:
        """ Saves raw tokenized sentences for training and all tokens with their
        corresponding senses for evaluation at 'corpus_cache_path'. """
        sentences = self.get_sentences()
        tagged_tokens = self.get_tagged_tokens()

        assert 'sentence' in sentences.columns
        assert 'token' in tagged_tokens.columns
        assert 'sense' in tagged_tokens.columns

        fh.save_df(self.corpus_cache_path,
                   fh.gen_sentences_file_name(self.corpus_name), sentences)
        fh.save_df(self.corpus_cache_path,
                   fh.gen_tagged_tokens_file_name(self.corpus_name),
                   tagged_tokens)
