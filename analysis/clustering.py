import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

import data.aggregator as da


def is_square_matrix(matrix: np.ndarray) -> bool:
    return len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]


def extract_sub_matrix(square_matrix: np.ndarray, indices: list) -> np.ndarray:
    """ Picks the entries from 'square_matrix' that 'indices' refers to. """
    assert is_square_matrix(square_matrix)
    selected_rows = square_matrix[indices]
    return selected_rows[:, indices]


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


def cluster_vectors_per_token(distance_matrix: np.ndarray, id_map: pd.DataFrame,
                              id_map_reduced: pd.DataFrame,
                              max_distance: float) -> pd.DataFrame:
    """ Clusters word-vectors per token, assigns a label for different meanings.
    Collects all references and word-vectors per token and label. """
    id_map['meaning_label'] = None

    for _, row in id_map_reduced.iterrows():
        sub_distance_matrix = extract_sub_matrix(distance_matrix,
                                                 row.word_vector_id)
        meaning = get_hierarchical_cluster(sub_distance_matrix, max_distance)
        id_map.loc[id_map.token == row.token, 'meaning_label'] = meaning

    return da.agg_references_and_word_vectors(id_map,
                                              by=['token', 'meaning_label'])
