import os
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from model.character_bert.character_bert import CharacterBertModel
from model.character_bert.character_cnn_utils import CharacterIndexer

import aggregation.aggregator as ag


def gen_model_cache_location(cache_directory: str, model_name: str) -> str:
    """ Generates the path for the model cache location. """
    return os.path.join(cache_directory, model_name)


def get_character_bert_from_cache(model_cache: str) -> CharacterBertModel:
    """ Loads a general CharacterBERT model from 'model_cache'. """
    model_path = gen_model_cache_location(model_cache, 'general_character_bert')
    return CharacterBertModel.from_pretrained(model_path)  # default: eval mode


def lower_tokens(tokenized_sentence: List[str]) -> List[str]:
    """ Transforms all tokens to lower case. """
    return [token.lower() for token in tokenized_sentence]


def add_special_tokens(tokenized_sentence: List[str]) -> List[str]:
    """ Wraps the 'tokenized_sentence' with special tokens. """
    return ['[CLS]', *tokenized_sentence, '[SEP]']


def encode_text(tokens: List[str], indexer: CharacterIndexer) -> torch.Tensor:
    """ Convert tokens into a padded tensor of character indices. """
    return indexer.as_padded_tensor([tokens])


def calc_word_vectors(encoded_sentence: torch.Tensor,
                      model: CharacterBertModel) -> torch.Tensor:
    """ Calculates the word vectors for 'encoded_sentence'. The output has shape
    [1, n, 768] for a sentence with n tokens. """
    with torch.no_grad():
        return model(encoded_sentence)[0]


def concat_word_vectors(word_vectors: List[torch.Tensor]) -> torch.Tensor:
    """ Concatenates word-vectors into a matrix. Each row is a word-vector. """
    assert all(len(v.shape) > 1 for v in word_vectors)

    return torch.cat(word_vectors, dim=0)


def strip_each(tokenized_sentences: List[Union[List[str], torch.Tensor]]) \
        -> List[Union[List[str], torch.Tensor]]:
    """ Drop the first and last token from every sentence or the first and last
    word vector in a matrix. Does not check if they are special tokens. """
    return [sentence[1:-1] for sentence in tokenized_sentences]


def add_special_tokens_to_each(tokenized_sentences: List[List[str]]) \
        -> List[List[str]]:
    """ Wraps each sentence with special tokens. """
    return [add_special_tokens(s) for s in tokenized_sentences]


def lower_sentences(tokenized_sentences: List[List[str]]) -> List[List[str]]:
    """ Lower cases all tokens in 'tokenized_sentences'. """
    return [lower_tokens(s) for s in tokenized_sentences]


def embed_sentences(tokenized_sentences: List[List[str]],
                    indexer: CharacterIndexer, model: CharacterBertModel) \
        -> Tuple[np.ndarray, pd.DataFrame]:
    """ Returns a matrix with one word-vector per token in each row, and a
    lookup table with reference-ids and word-vector-ids per token. """
    encoded_sentences = [encode_text(s, indexer) for s in tokenized_sentences]
    word_vectors = [calc_word_vectors(e, model).squeeze(dim=0) for e in
                    encoded_sentences]

    tokenized_sentences = strip_each(tokenized_sentences)
    word_vectors = strip_each(word_vectors)

    word_vectors = concat_word_vectors(word_vectors)
    id_map = ag.gen_ids_for_sentences_and_tokens(tokenized_sentences)

    return word_vectors.numpy(), id_map
