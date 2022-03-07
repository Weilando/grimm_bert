import pandas as pd
import torch
from transformers import BertTokenizer, BatchEncoding, BertModel


def tokenize_text_log(text: str, tokenizer: BertTokenizer) -> None:
    tokenized_text = encode_text(text, tokenizer)
    print(f"Tokenized input: {tokenizer.tokenize(text)}")
    print(f"Input tensor: {tokenized_text.input_ids}")


def encode_text(text: str, tokenizer: BertTokenizer) -> BatchEncoding:
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
