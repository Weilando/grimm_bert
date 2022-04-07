import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from analysis.linkage_name import LinkageName


def get_hierarchical_cluster(word_vectors: np.ndarray, token: str,
                             linkage_name: LinkageName, max_distance: float) \
        -> np.ndarray:
    """ Generates a hierarchical clustering based on cosine distances of rows
    from 'word_vectors'. Vectors within 'max_distance' form one cluster. Names
    the clusters according to 'token'. """
    if word_vectors.shape[0] < 2:
        return np.array([f"{token}_0"])

    sense_ids = AgglomerativeClustering(n_clusters=None,
                                        affinity='cosine',
                                        linkage=linkage_name,
                                        distance_threshold=max_distance) \
        .fit_predict(word_vectors)

    return np.array([f"{token}_{sense_id}" for sense_id in sense_ids])


def cluster_vectors_per_token(word_vectors: np.ndarray,
                              id_map_reduced: pd.DataFrame,
                              linkage_name: LinkageName, max_distance: float) \
        -> pd.DataFrame:
    """ Clusters word-vectors per token based on their cosine distances and adds
    unique labels for senses. """
    id_map_reduced['sense'] = id_map_reduced.apply(
        lambda r: get_hierarchical_cluster(word_vectors[r.word_vector_id],
                                           r.token, linkage_name, max_distance),
        axis=1)

    return id_map_reduced
