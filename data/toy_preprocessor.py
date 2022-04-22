import pandas as pd

from data.corpus_handler import CorpusName
from data.corpus_preprocessor import CorpusPreprocessor
from grimm_bert import DEFAULT_CORPUS_CACHE_DIR

SENTENCES = [['She', 'glances', 'at', 'her', 'watch', '.'],
             ['He', 'wants', 'to', 'watch', 'the', 'soccer', 'match', '.'],
             ['We', 'watch', 'movies', 'and', 'eat', 'chips', '.'],
             ['The', 'watch', 'should', 'match', 'her', 'coat', '.'],
             ['The', 'new', 'chips', 'are', 'twice', 'as', 'fast', '.'],
             ['She', 'wears', 'a', 'coat', 'to', 'keep', 'warm', '.'],
             ['He', 'earned', 'many', 'chips', 'in', 'this', 'game', '.'],
             ['He', 'earned', 'angry', 'glances', '.']]

TOKENS = ['she', 'glances', 'at', 'her', 'watch', '.',
          'he', 'wants', 'to', 'watch', 'the', 'soccer', 'match', '.',
          'we', 'watch', 'movies', 'and', 'eat', 'chips', '.',
          'the', 'watch', 'should', 'match', 'her', 'coat', '.',
          'the', 'new', 'chips', 'are', 'twice', 'as', 'fast', '.',
          'she', 'wears', 'a', 'coat', 'to', 'keep', 'warm', '.',
          'he', 'earned', 'many', 'chips', 'in', 'this', 'game', '.',
          'he', 'earned', 'angry', 'glances', '.']

SENSES = ['she0', 'glances0', 'at0', 'her0', 'watch0', '.0',
          'he0', 'wants0', 'to0', 'watch1', 'the0', 'soccer0', 'match0', '.0',
          'we0', 'watch1', 'movies0', 'and0', 'eat0', 'chips0', '.0',
          'the0', 'watch0', 'should0', 'match1', 'her0', 'coat0', '.0',
          'the0', 'new0', 'chips1', 'are0', 'twice0', 'as0', 'fast0', '.0',
          'she0', 'wears0', 'a0', 'coat0', 'to0', 'keep0', 'warm0', '.0',
          'he0', 'earned0', 'many0', 'chips3', 'in0', 'this0', 'game0', '.0',
          'he0', 'earned1', 'angry0', 'glances1', '.0']


class ToyPreprocessor(CorpusPreprocessor):
    def __init__(self, corpus_name: CorpusName = CorpusName.TOY,
                 corpus_cache_path: str = DEFAULT_CORPUS_CACHE_DIR):
        super().__init__(corpus_name, corpus_cache_path)

    def get_sentences(self) -> pd.DataFrame:
        return pd.DataFrame({'sentence': SENTENCES})

    def get_tagged_tokens(self) -> pd.DataFrame:
        return pd.DataFrame({'token': TOKENS, 'sense': SENSES})


if __name__ == '__main__':
    toy_preprocessor = ToyPreprocessor()
    toy_preprocessor.cache_dataset()
