import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

import data.aggregator as da


def calc_linkage_matrix(model: AgglomerativeClustering) -> np.ndarray:
    """ Counts the samples under each node. """
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    return np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)


def get_hierarchical_cluster(distance_matrix: np.ndarray, max_distance: float)\
        -> AgglomerativeClustering:
    """ Generates a hierarchical clustering for rows of 'distance_matrix'.
    'max_dist' is the maximum allowed distance for the same cluster. """
    algo = AgglomerativeClustering(n_clusters=None, compute_distances=False,
                                   affinity='precomputed', linkage='single',
                                   distance_threshold=max_distance)
    algo.fit(distance_matrix)
    return algo


def cluster_feature(feature: str, distance_matrix: np.ndarray,
                    id_map: pd.DataFrame, id_map_reduced: pd.DataFrame) \
        -> pd.DataFrame:
    """ Clustering with consideration of 'feature' per token. """
    assert feature in id_map
    id_map['meaning'] = None

    for _, row in id_map_reduced.iterrows():
        ids = row[feature]
        distance_matrix_rows = distance_matrix[ids]
        sub_distance_matrix = distance_matrix_rows[:, ids]
        if sub_distance_matrix.shape[1] < 2:
            meaning = 0
        else:
            cl = get_hierarchical_cluster(sub_distance_matrix,
                                          max_distance=0.1)
            meaning = cl.labels_
        id_map.loc[id_map.token == row.token, 'meaning'] = meaning

    return da.collect_references_and_word_vectors(id_map,
                                                  by=['token', 'meaning'])


def cluster_general(distance_matrix: np.ndarray, id_map: pd.DataFrame) \
        -> pd.DataFrame:
    """ Clustering without consideration of tokens or references. """
    cluster = get_hierarchical_cluster(distance_matrix, max_distance=0.1)
    id_map['cluster'] = cluster.labels_
    return id_map
