import os
from typing import List, Union

import pandas as pd
from nltk.corpus import SemcorCorpusReader, semcor
from nltk.tree.tree import Tree

import data.file_handler as fh

STD_SENSE = '_SENSE'
SENTENCES_NAME = 'semcor_sentences'
TAGGED_TOKENS_NAME = 'semcor_tagged_tokens'


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


def cache_semcor_dataset(absolute_path: os.path):
    """ Saves raw sentences for training and all tokens with their corresponding
    senses for evaluation at 'absolute_path'. """
    sentences = pd.DataFrame({'sentence': semcor.sents()})
    tagged_sentences = get_sentences_with_sense_tags(semcor)
    tagged_tokens = get_tokens_and_senses_from_sentences(tagged_sentences)

    fh.save_df(absolute_path, fh.gen_df_file_name(SENTENCES_NAME), sentences)
    fh.save_df(absolute_path, fh.gen_df_file_name(TAGGED_TOKENS_NAME),
               tagged_tokens)
