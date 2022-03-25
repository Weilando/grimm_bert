import os
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from model.character_bert import CharacterBertModel
from model.character_cnn_utils import CharacterIndexer
from transformers import BertTokenizer

from analysis import aggregator as ag


def gen_model_cache_location(cache_directory: str, model_name: str) -> str:
    """ Generates the path for the model cache location. """
    return os.path.join(cache_directory, model_name)


def tokenize_text(text: str, tokenizer: BertTokenizer) -> List[str]:
    """ Tokenize 'text' and add the [CLS] and [SEP] token. """
    tokenized_text = tokenizer.basic_tokenizer.tokenize(text)
    return ['[CLS]', *tokenized_text, '[SEP]']


def encode_text(tokens: List[str], indexer: CharacterIndexer) -> torch.Tensor:
    """ Convert tokens into a padded tensor of character indices. """
    return indexer.as_padded_tensor([tokens])


def calc_word_vectors(encoded_text: torch.Tensor, model: CharacterBertModel) \
        -> torch.Tensor:
    """ Calculates the word vectors for 'encoded_text'. The output has shape
    [1, n, 768] for a text with n tokens. """
    with torch.no_grad():
        return model(encoded_text)[0]


def parse_sentences(sentences: List[str], tokenizer: BertTokenizer,
                    indexer: CharacterIndexer, model: CharacterBertModel) \
        -> Tuple[np.ndarray, pd.DataFrame]:
    """ Parses 'sentences' with 'tokenizer'. Returns a tensor with one
    word-vector per token in each row, and a lookup table with reference-ids and
    word-vector-ids per token. """
    tokenized_sentences = [tokenize_text(s, tokenizer) for s in sentences]
    encoded_sentences = [encode_text(s, indexer) for s in tokenized_sentences]
    word_vectors = [calc_word_vectors(e, model).squeeze(dim=0) for e in
                    encoded_sentences]

    tokenized_sentences = strip_sentences(tokenized_sentences)
    word_vectors = strip_sentences(word_vectors)

    word_vectors = ag.concat_word_vectors(word_vectors)
    id_map = ag.gen_ids_for_vectors_and_references(tokenized_sentences)

    return word_vectors.numpy(), id_map


def strip_sentences(sentences: List[Union[List[str], torch.Tensor]]) \
        -> List[Union[List[str], torch.Tensor]]:
    """ Drop the first and last token from every sentence or the first and last
    word vector in a matrix. Does not check if they are special tokens. """
    return [sentence[1:-1] for sentence in sentences]
