from data.corpus_name import CorpusName


def gen_dictionary_file_name(experiment_prefix: str) -> str:
    """ Generates a file name for a dictionary DataFrame. """
    return f"{experiment_prefix}-dictionary.pkl"


def gen_html_dictionary_file_name(experiment_prefix: str) -> str:
    """ Generates a file name for a dictionary HTML file. """
    return f"{experiment_prefix}-dictionary.html"


def gen_raw_id_map_file_name(corpus: CorpusName) -> str:
    """ Generates a file name for a raw id_map DataFrame. """
    return f"{corpus.lower()}-raw_id_map.pkl"


def gen_sentences_file_name(corpus: CorpusName) -> str:
    """ Generates a file name for a DataFrame with sentences. """
    return f"{corpus.lower()}-sentences.pkl"


def gen_stats_file_name(experiment_prefix: str) -> str:
    """ Generates a file name for a dict with experiment statistics. """
    return f"{experiment_prefix}-stats.json"


def gen_tagged_tokens_file_name(corpus: CorpusName) -> str:
    """ Generates a file name for a DataFrame with tokens and tags. """
    return f"{corpus.lower()}-tagged_tokens.pkl"


def gen_word_vec_file_name(corpus: CorpusName) -> str:
    """ Generates a file name for a word vector matrix. """
    return f"{corpus.lower()}-word_vectors.npz"
