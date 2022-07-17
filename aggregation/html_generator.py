from typing import List

import pandas as pd

import aggregation.aggregator as ag


def gen_html_start(experiment_name: str) -> str:
    """ Generates the beginning of an HTML file, including a head section and
    the beginning of the body. 'experiment' is the page title. """
    return f"<!DOCTYPE html>\n<html>\n" \
           f"<head>\n<title>{experiment_name}</title>\n</head>\n" \
           f"<body>\n<h1>{experiment_name}</h1>\n"


def gen_html_end() -> str:
    """ Generates the end of an HTML file, closing the body and html tag. """
    return "</body>\n</html>\n"


def gen_token_listing(dictionary: pd.DataFrame) -> str:
    """ Generates a listing of all token in the dictionary. Each token links to
    its entry in the thesaurus. """
    assert 'token' in dictionary.columns
    return '<hr>\n<h2>Tokens</h2>\n' \
           + ',\n'.join(f'<a href="#{t}">{t}</a>' for t in dictionary.token) \
           + "\n"


def gen_token_thesaurus(dictionary: pd.DataFrame) -> str:
    """ Generates a thesaurus with one entry per token. Each entry lists all
    corresponding senses and refers to sentences from the corpus. """
    assert 'token' in dictionary.columns
    entries = ['<hr>\n<h2>Thesaurus</h2>']
    last_token = None
    for row in dictionary.itertuples():
        if last_token != row.token:
            if last_token is not None:
                entries.append('</ul>')
            last_token = row.token
            entries.append(f'<hr>\n<h3 id="{row.token}">{row.token}</h3>\n<ul>')
        entries.append(f'<li id="{row.sense}"><b>{row.sense}: </b>')
        entries.append(', '.join(
            f'<a href="#{s}">Sentence {s}</a>' for s in row.sentence_id))
        entries.append('</li>')
    return '\n'.join(entries) + '</ul>\n'


def gen_sentence_listing(sentences: List[List[str]]):
    """ Lists all sentences and adds the index as id. Requires a list of
    sentences, where each sentence is a list of tokens. """
    sentence_listing = [
        f'<li><b id={s_id}>{s_id}: </b>'
        + ' '.join(s)
        + '</li>' for s_id, s in enumerate(sentences)]
    return '<hr>\n<h2>Sentences</h2>\n<ul>\n' \
           + '\n'.join(sentence_listing) \
           + '\n</ul>\n'


def render_dictionary_in_html(dictionary: pd.DataFrame,
                              sentences: List[List[str]],
                              experiment_name: str) -> str:
    """ Renders the given dictionary as an HTML file. """
    dictionary_sense_level = ag.pack_sentence_ids_and_token_ids(
        ag.unpack_and_sort_per_token_id(
            dictionary,
            ['sentence_id', 'token_id', 'sense']),
        ['token', 'sense'])
    return gen_html_start(experiment_name) + gen_token_listing(
        dictionary) + gen_token_thesaurus(
        dictionary_sense_level) + gen_sentence_listing(
        sentences) + gen_html_end()
