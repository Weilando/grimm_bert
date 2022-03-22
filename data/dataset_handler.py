import pandas as pd
from nltk.corpus.reader import SemcorCorpusReader
from nltk.corpus.reader.wordnet import Lemma
from nltk.tree.tree import Tree

STD_SENSE = 'STD_SENSE'


def extract_tokens_and_senses_from_list(word_list: list) -> pd.DataFrame:
    return pd.DataFrame(data={'token': word_list,
                              'sense': STD_SENSE})


def extract_tokens_and_senses_from_tree(tree: Tree) -> pd.DataFrame:
    assert isinstance(tree.label(), Lemma) or isinstance(tree.label(), str)
    assert isinstance(tree.leaves(), list)
    sense = tree.label().name() if isinstance(tree.label(), Lemma) \
        else tree.label()
    return pd.DataFrame(data={'token': tree.leaves(),
                              'sense': sense})


def extract_tokens_and_senses_from_sentence(sentence: list) -> pd.DataFrame:
    tokens_and_senses = [(extract_tokens_and_senses_from_tree(element)
                          if isinstance(element, Tree) else
                          extract_tokens_and_senses_from_list(element))
                         for element in sentence]
    return pd.concat(tokens_and_senses, ignore_index=True)


def extract_tokens_and_senses_from_sentences(sentences: list) -> pd.DataFrame:
    tokens_and_senses = [extract_tokens_and_senses_from_sentence(sentence)
                         for sentence in sentences]
    return pd.concat(tokens_and_senses, ignore_index=True)


def get_tagged_sentences(corpus_reader: SemcorCorpusReader) -> list:
    return corpus_reader.tagged_sents(tag='sem')
