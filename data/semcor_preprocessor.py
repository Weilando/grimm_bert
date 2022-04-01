from typing import List, Union

import pandas as pd
from nltk.corpus import SemcorCorpusReader, semcor
from nltk.tree.tree import Tree

from data.corpus_handler import CorpusName
from data.corpus_preprocessor import CorpusPreprocessor
from grimm_bert import DEFAULT_CORPUS_CACHE_DIR

STD_SENSE = '_SENSE'


def get_tokens_and_senses_from_list(tokens: List[str], sense: str = STD_SENSE) \
        -> pd.DataFrame:
    """ Assigns generated, lower cased sense names to a list of tokens. """
    senses = [f"{token.lower()}{sense}" for token in tokens]
    return pd.DataFrame({'token': tokens, 'sense': senses})


def get_tokens_and_senses_from_tree(tree: Tree) -> pd.DataFrame:
    """ Assigns the corresponding sense per token. Tokens are the leaves of
    'tree' and the common sense is its root label. The label can be either a str
    or a nltk.corpus.wordnet.Lemma and is always represented as str. Lower cases
    and prepends the token itself to ensure unique senses across tokens. """
    tokens = tree.leaves()
    senses = [f"{token.lower()}_{tree.label()}" for token in tokens]
    return pd.DataFrame({'token': tokens, 'sense': senses})


def get_tokens_and_senses_from_sentence(sentence: List[Union[Tree, List[str]]],
                                        sense: str = STD_SENSE) -> pd.DataFrame:
    """ Extracts tokens and their respective senses from 'sentence'. The sense
    can be either a str or a lemma from WordNet. Lower cases and prepends the
    token itself to ensure unique senses across tokens. """
    tokens_and_senses = [(get_tokens_and_senses_from_tree(element)
                          if isinstance(element, Tree) else
                          get_tokens_and_senses_from_list(element, sense))
                         for element in sentence]
    return pd.concat(tokens_and_senses, ignore_index=True)


def get_tokens_and_senses_from_sentences(
        sentences: List[List[Union[Tree, List[str]]]], sense: str = STD_SENSE) \
        -> pd.DataFrame:
    """ Extracts all tokens and their respective senses from sentences in
    'sentences'. The sense can be either a str or a lemma from WordNet. Prepends
    the token itself to ensure unique senses across tokens."""
    tokens_and_senses = [get_tokens_and_senses_from_sentence(sentence, sense)
                         for sentence in sentences]
    return pd.concat(tokens_and_senses, ignore_index=True)


def get_sentences_with_sense_tags(corpus_reader: SemcorCorpusReader) -> List:
    """ Returns a list of chunked, tokenized sentences with semantic tags. """
    return corpus_reader.tagged_sents(tag='sem')


class SemcorPreprocessor(CorpusPreprocessor):
    def __init__(self, corpus_name: CorpusName = CorpusName.SEMCOR,
                 corpus_cache_path: str = DEFAULT_CORPUS_CACHE_DIR):
        super().__init__(corpus_name, corpus_cache_path)

    def get_sentences(self) -> pd.DataFrame:
        return pd.DataFrame({'sentence': semcor.sents()})

    def get_tagged_tokens(self) -> pd.DataFrame:
        tagged_sentences = get_sentences_with_sense_tags(semcor)
        return get_tokens_and_senses_from_sentences(tagged_sentences)


if __name__ == '__main__':
    semcor_preprocessor = SemcorPreprocessor()
    semcor_preprocessor.cache_dataset()
