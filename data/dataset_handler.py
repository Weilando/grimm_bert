from typing import List, Union

import pandas as pd
from nltk.corpus.reader import SemcorCorpusReader
from nltk.tree.tree import Tree

STD_SENSE = '_SENSE'

TOY_SENTENCES = ["He wears a watch.", "She glances at her watch.",
                 "He wants to watch the soccer match."]

TOY_SENSES = [0, 1, 2, 3, 4,
              5, 6, 7, 8, 3, 4,
              0, 9, 10, 11, 12, 13, 14, 4]


def get_tokens_and_senses_from_list(tokens: List[str], sense: str = STD_SENSE) \
        -> pd.DataFrame:
    """ Assigns generated sense names to a list of tokens. """
    senses = [f"{token}{sense}" for token in tokens]
    return pd.DataFrame({'token': tokens, 'sense': senses})


def get_tokens_and_senses_from_tree(tree: Tree) -> pd.DataFrame:
    """ Assigns the corresponding sense per token. Tokens are the leaves of
    'tree' and the common sense is its root label. The label can be either a str
    or a nltk.corpus.wordnet.Lemma and is always represented as str. Prepends
    the token itself to ensure unique senses across tokens. """
    tokens = tree.leaves()
    senses = [f"{token}_{tree.label()}" for token in tokens]
    return pd.DataFrame({'token': tokens, 'sense': senses})


def get_tokens_and_senses_from_sentence(sentence: List[Union[Tree, List[str]]],
                                        sense: str = STD_SENSE) -> pd.DataFrame:
    """ Extracts tokens and their respective senses from 'sentence'. The sense
    can be either a str or a lemma from WordNet. Prepends the token itself to
    ensure unique senses across tokens. """
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
    """ Returns a list of chunked sentences with semantic tags. """
    return corpus_reader.tagged_sents(tag='sem')
