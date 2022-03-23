import logging
import os
from typing import List

import pandas as pd
import torch
from transformers import BertTokenizer, BatchEncoding, BertModel

import data.aggregator as da


def should_cache_model(model_cache_location: str) -> bool:
    """ Indicates if the model cache is not present yet. """
    return not os.path.exists(model_cache_location)


def gen_model_cache_location(cache_directory: str, model_name: str) -> str:
    """ Generates the path for the model cache location. """
    return os.path.join(cache_directory, model_name)


def cache_model(tokenizer: BertTokenizer, model: BertModel,
                cache_location: str) -> None:
    """ Caches the Huggingface model and tokenizer at 'cache_location'. """
    tokenizer.save_pretrained(cache_location)
    model.save_pretrained(cache_location)
    logging.info(f"Cached model and tokenizer at {cache_location}.")


def encode_text(text: str, tokenizer: BertTokenizer) -> BatchEncoding:
    """ Tokenizes 'text' with 'tokenizer' as torch.tensors. """
    return tokenizer(text, return_tensors='pt')


def add_decoded_tokens(df: pd.DataFrame, tokenizer: BertTokenizer) \
        -> pd.DataFrame:
    """ Adds a column with decoded tokens to 'df' using 'tokenizer'. """
    df['decoded_token'] = [tokenizer.decode([t]) for t in df.token]
    return df


def calc_word_vectors(encoded_text: BatchEncoding, model: BertModel) \
        -> torch.tensor:
    """ Calculates the word vectors for 'encoded_text'. The output has shape
    [1, n, 768] for BERT and a text with n tokens. """
    with torch.no_grad():
        return model(**encoded_text, output_hidden_states=True) \
            .last_hidden_state


def parse_sentences(sentences: List[str], tokenizer: BertTokenizer,
                    model: BertModel) -> tuple:
    """ Parses 'sentences' with 'tokenizer'. Returns a tensor with one
    word-vector per token in each row, and a lookup table with reference-ids and
    word-vector-ids per token. """
    encoded_sentences = [encode_text(s, tokenizer) for s in sentences]
    word_vectors = [calc_word_vectors(e, model).squeeze(dim=0) for e in
                    encoded_sentences]

    word_vectors = da.concat_word_vectors(word_vectors)
    id_map = da.gen_ids_for_vectors_and_references(encoded_sentences)

    return word_vectors.numpy(), id_map
