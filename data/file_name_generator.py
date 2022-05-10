from clustering.linkage_name import LinkageName
from clustering.metric_name import MetricName
from data.corpus_name import CorpusName


def gen_dictionary_file_name(experiment_prefix: str) -> str:
    """ Generates a file name for a dictionary DataFrame. """
    return f"{experiment_prefix}-dictionary.pkl"


def gen_experiment_prefix(corpus: CorpusName, affinity: MetricName,
                          linkage: LinkageName, distance: float) -> str:
    """ Generates a prefix for experiment result files. """
    return f"{corpus.lower()}-affinity_{affinity.lower()}-" \
           f"linkage_{linkage.lower()}-dist_{distance}"


def gen_experiment_prefix_no_dist(corpus: CorpusName, affinity: MetricName,
                                  linkage: LinkageName) -> str:
    """ Generates a prefix for experiment result files. """
    return f"{corpus.lower()}-affinity_{affinity.lower()}" \
           f"-linkage_{linkage.lower()}-no_dist"


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
