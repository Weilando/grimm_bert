import os

import pandas as pd

import data.file_handler as fh

SENTENCES_NAME = 'toy_sentences'
TAGGED_TOKENS_NAME = 'toy_tagged_tokens'

SENTENCES = ["He wears a watch.", "She glances at her watch.",
             "He wants to watch the soccer match."]

TOKENS = ['he', 'wears', 'a', 'watch', '.',
          'she', 'glances', 'at', 'her', 'watch', '.',
          'he', 'wants', 'to', 'watch', 'the', 'soccer', 'match', '.']

SENSES = [0, 1, 2, 3, 4,
          5, 6, 7, 8, 3, 4,
          0, 9, 10, 11, 12, 13, 14, 4]


def cache_toy_dataset(absolute_path: os.path):
    """ Saves raw sentences for training and all tokens with their corresponding
    senses for evaluation at 'absolute_path'. """
    sentences = pd.DataFrame({'sentence': SENTENCES})
    tagged_tokens = pd.DataFrame({'token': TOKENS, 'sense': SENSES})

    fh.save_df(absolute_path, fh.gen_df_file_name(SENTENCES_NAME), sentences)
    fh.save_df(absolute_path, fh.gen_df_file_name(TAGGED_TOKENS_NAME),
               tagged_tokens)
