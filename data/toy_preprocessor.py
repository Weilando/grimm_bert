from typing import List

import pandas as pd

from data.corpus_handler import CorpusName
from data.corpus_preprocessor import CorpusPreprocessor
from grimm_bert import DEFAULT_CORPUS_CACHE_DIR

SENTENCES = [
    ['she', 'glances', 'at', 'her', 'watch', '.'],
    ['we', 'want', 'to', 'watch', 'some', 'pool', 'matches', '.'],
    ['we', 'watch', 'movies', 'and', 'eat', 'chips', '.'],
    ['can', 'you', 'please', 'watch', 'the', 'kids', '?'],
    ['the', 'scarf', 'matches', 'her', 'coat', '.'],
    ['the', 'last', 'chips', 'read', 'twice', 'as', 'fast', '.'],
    ['she', 'wears', 'a', 'warm', 'coat', 'even', 'in', 'may', '.'],
    ['he', 'earned', 'some', 'warm', 'glances', '.'],
    ['some', 'earned', 'money', 'with', 'cold', 'soda', '.'],
    ['can', 'we', 'book', 'a', 'hotel', 'with', 'a', 'pool', '?'],
    ['do', 'you', 'read', 'her', 'last', 'book', '?'],
    ['may', 'he', 'drink', 'the', 'last', 'can', 'of', 'soda', '?'],
    ['he', 'wears', 'a', 'scarf', 'as', 'he', 'has', 'a', 'cold', '.'],
    ['we', 'fast', 'soda', 'and', 'chips', '.'],
    ['are', 'we', 'even', '?'],
    ['we', 'can', 'eat', 'some', 'chips', 'at', 'the', 'movies', '.'],
    ['the', 'book', 'is', 'a', 'fun', 'read', '.'],
    ['the', 'kids', 'want', 'some', 'fun', 'in', 'the', 'pool', '.'],
    ['may', 'we', 'order', 'a', 'drink', '?'],
    ['please', 'watch', 'the', 'movies', 'in', 'order', '.']]

SENSES = [
    'she0', 'glances0', 'at0', 'her0', 'watch0', '.0',
    'we0', 'want0', 'to0', 'watch1', 'some0', 'pool0', 'matches0', '.0',
    'we0', 'watch1', 'movies0', 'and0', 'eat0', 'chips0', '.0',
    'can0', 'you0', 'please0', 'watch2', 'the0', 'kids0', '?0',
    'the0', 'scarf0', 'matches1', 'her0', 'coat0', '.0',
    'the0', 'last0', 'chips1', 'read0', 'twice0', 'as0', 'fast0', '.0',
    'she0', 'wears0', 'a0', 'warm0', 'coat0', 'even0', 'in0', 'may0', '.0',
    'he0', 'earned0', 'some0', 'warm1', 'glances1', '.0',
    'some1', 'earned1', 'money0', 'with0', 'cold0', 'soda0', '.',
    'can0', 'we0', 'book0', 'a0', 'hotel0', 'with1', 'a0', 'pool1', '?0',
    'do0', 'you0', 'read1', 'her0', 'last0', 'book1', '?0',
    'may1', 'he0', 'drink0', 'the0', 'last1', 'can1', 'of0', 'soda0', '?0',
    'he0', 'wears0', 'a0', 'scarf0', 'as1', 'he0', 'has0', 'a0', 'cold1', '.0',
    'we0', 'fast1', 'soda0', 'and0', 'chips0', '.0',
    'are0', 'we0', 'even1', '?0',
    'we0', 'can0', 'eat0', 'some0', 'chips0', 'at1', 'the0', 'movies1', '.0',
    'the0', 'book1', 'is0', 'a0', 'fun0', 'read2', '.0',
    'the0', 'kids0', 'want0', 'some0', 'fun1', 'in1', 'the0', 'pool1', '.0',
    'may1', 'we0', 'order0', 'a0', 'drink1', '?0',
    'please0', 'watch1', 'the0', 'movies0', 'in2', 'order1', '.0']


def flatten_list(nested_list: List[List[str]]) -> List[str]:
    return sum(nested_list, [])


class ToyPreprocessor(CorpusPreprocessor):
    def __init__(self, corpus_name: CorpusName = CorpusName.TOY,
                 corpus_cache_path: str = DEFAULT_CORPUS_CACHE_DIR):
        super().__init__(corpus_name, corpus_cache_path)

    def get_sentences(self) -> pd.DataFrame:
        return pd.DataFrame({'sentence': SENTENCES})

    def get_tagged_tokens(self) -> pd.DataFrame:
        return pd.DataFrame({'token': flatten_list(SENTENCES),
                             'sense': SENSES, 'tagged_sense': True})


if __name__ == '__main__':
    toy_preprocessor = ToyPreprocessor()
    toy_preprocessor.cache_dataset()
