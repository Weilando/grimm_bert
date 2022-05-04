from pandas import DataFrame

from data.corpus_handler import CorpusName
from data.file_handler import add_and_get_abs_path, save_df
from data.file_name_generator import gen_sentences_file_name, \
    gen_tagged_tokens_file_name

STD_SENSE = '_SENSE'


class CorpusPreprocessor(object):
    def __init__(self, corpus_name: CorpusName, corpus_cache_path: str):
        self.corpus_name = corpus_name
        self.corpus_cache_path = add_and_get_abs_path(corpus_cache_path)

    def get_sentences(self) -> DataFrame:
        """ Returns a DataFrame with a column 'sentence', where each sentence is
        a list of tokens. """
        pass

    def get_tagged_tokens(self) -> DataFrame:
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
        assert 'tagged_sense' in tagged_tokens.columns

        save_df(self.corpus_cache_path,
                gen_sentences_file_name(self.corpus_name), sentences)
        save_df(self.corpus_cache_path,
                gen_tagged_tokens_file_name(self.corpus_name),
                tagged_tokens)
