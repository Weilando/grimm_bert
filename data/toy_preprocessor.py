import pandas as pd

from data.corpus_handler import CorpusName
from data.corpus_preprocessor import CorpusPreprocessor
from grimm_bert import DEFAULT_CORPUS_CACHE_DIR

SENTENCES = [['He', 'wears', 'a', 'watch', '.'],
             ['She', 'glances', 'at', 'the', 'watch', 'often', '.'],
             ['He', 'wants', 'to', 'watch', 'the', 'soccer', 'match', '.'],
             ['We', 'watch', 'movies', 'and', 'eat', 'popcorn', '.']]

TOKENS = ['he', 'wears', 'a', 'watch', '.',
          'she', 'glances', 'at', 'the', 'watch', 'often', '.',
          'he', 'wants', 'to', 'watch', 'the', 'soccer', 'match', '.',
          'we', 'watch', 'movies', 'and', 'eat', 'popcorn', '.']

SENSES = ['he0', 'wears0', 'a0', 'watch0', '.0',
          'she0', 'glances0', 'at0', 'the0', 'watch0', 'often0', '.0',
          'he0', 'wants0', 'to0', 'watch1', 'the0', 'soccer0', 'match0', '.0',
          'we0', 'watch1', 'movies0', 'and0', 'eat0', 'popcorn0', '.0']


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
