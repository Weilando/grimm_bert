import logging
import os

import pandas as pd
import torch
from transformers import BertTokenizer, BatchEncoding, BertModel


def should_download_model(model_name: str, model_cache_path: str) -> bool:
    """ Indicates if the model cache is not present yet. """
    return not os.path.exists(os.path.join(model_name, model_cache_path))


def download_and_cache_model(model_name: str, model_cache_path: str) -> None:
    """ Downloads the Huggingface model and tokenizer for 'model_name' and saves
    them in directory 'model_name' at 'cache_path'. """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    cache_location = os.path.join(model_cache_path, model_name)
    tokenizer.save_pretrained(cache_location)
    model.save_pretrained(cache_location)
    logging.info(f"Cached model and tokenizer at {model_cache_path}.")


def encode_text(text: str, tokenizer: BertTokenizer) -> BatchEncoding:
    """ Tokenizes 'text' with 'tokenizer' as torch.tensors. """
    return tokenizer(text, return_tensors='pt')


def add_decoded_tokens(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """ Adds a column with decoded tokens to 'df' using the tokenizer for
    'model_name'. """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    df['decoded_token'] = [tokenizer.decode([t]) for t in df.token]
    return df


def calc_word_vectors(encoded_text: BatchEncoding, model: BertModel) \
        -> torch.tensor:
    """ Calculates the word vectors for 'encoded_text'. The output has shape
    [1, n, 768] for BERT and a text with n tokens. """
    with torch.no_grad():
        return model(**encoded_text, output_hidden_states=True) \
            .last_hidden_state
