from typing import List

import pandas as pd
from nltk.corpus.reader import SemcorCorpusReader
from nltk.tree.tree import Tree

STD_SENSE = 'STD_SENSE'


def extract_tokens_and_senses_from_list(word_list: List[str]) -> pd.DataFrame:
    """ Assigns the standard sense to a list of tokens. """
    return pd.DataFrame({'token': word_list, 'sense': STD_SENSE})


def extract_tokens_and_senses_from_tree(tree: Tree) -> pd.DataFrame:
    """ Assigns the corresponding sense per token. Tokens are the leaves of
    'tree' and the common sense is its root label. The label can be either a str
    or a nltk.corpus.wordnet.Lemma and is always represented as str. """
    return pd.DataFrame({'token': tree.leaves(), 'sense': f"{tree.label()}"})


def extract_tokens_and_senses_from_sentence(sentence: list) -> pd.DataFrame:
    """ Extracts tokens and their respective senses from 'sentence'. The sense
    can be either a str or a lemma from WordNet. """
    tokens_and_senses = [(extract_tokens_and_senses_from_tree(element)
                          if isinstance(element, Tree) else
                          extract_tokens_and_senses_from_list(element))
                         for element in sentence]
    return pd.concat(tokens_and_senses, ignore_index=True)


def extract_tokens_and_senses_from_sentences(sentences: list) -> pd.DataFrame:
    """ Extracts all tokens and their respective senses from sentences in
    'sentences'. The sense can be either a str or a lemma from WordNet. """
    tokens_and_senses = [extract_tokens_and_senses_from_sentence(sentence)
                         for sentence in sentences]
    return pd.concat(tokens_and_senses, ignore_index=True)


def get_sentences_with_sense_tags(corpus_reader: SemcorCorpusReader) -> list:
    """ Returns a list of sentences with semantic tags from 'corpus_reader'. """
    return corpus_reader.tagged_sents(tag='sem')
