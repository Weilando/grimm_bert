from typing import List

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

from analysis import aggregator as ag


def is_square_matrix(matrix: np.ndarray) -> bool:
    return len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]


def get_hierarchical_cluster(distance_matrix: np.ndarray, max_distance: float) \
        -> np.ndarray:
    """ Generates a hierarchical clustering for rows of 'distance_matrix'.
    'max_dist' is the maximum allowed distance for the same cluster. """
    assert is_square_matrix(distance_matrix)
    if distance_matrix.shape[0] < 2:
        return np.array([0])

    return AgglomerativeClustering(n_clusters=None, compute_distances=False,
                                   affinity='precomputed', linkage='single',
                                   distance_threshold=max_distance) \
        .fit_predict(distance_matrix)


def cluster_vectors_per_token(word_vectors: np.ndarray, id_map: pd.DataFrame,
                              max_distance: float) -> pd.DataFrame:
    """ Clusters word-vectors per token based on their cosine distances and adds
    unique labels for senses. """
    id_map['sense'] = None

    id_map_reduced = ag.agg_references_and_word_vectors(id_map, 'token')
    for _, row in id_map_reduced.iterrows():
        distance_matrix = cosine_distances(word_vectors[row.word_vector_id])
        sense_ids = get_hierarchical_cluster(distance_matrix, max_distance)
        senses = [f"{row.token}_{sense_id}" for sense_id in sense_ids]
        id_map.loc[id_map.token == row.token, 'sense'] = senses

    return id_map


def reduce_dictionary(dictionary: pd.DataFrame) -> pd.DataFrame:
    """ Collects and sorts references and word vectors per token and sense. """
    return ag.agg_references_and_word_vectors(dictionary, by=['token', 'sense'])


def extract_int_senses(dictionary: pd.DataFrame) -> List[int]:
    """ Enumerates unique senses and returns an array of those sense ids. """
    return dictionary.sense.factorize()[0].tolist()
