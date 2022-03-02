import numpy as np
from sklearn.cluster import AgglomerativeClustering


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
