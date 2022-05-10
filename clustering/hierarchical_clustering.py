import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from clustering.linkage_name import LinkageName
from clustering.metric_name import MetricName


def get_clusters_for_token(word_vectors: np.ndarray, token: str,
                           affinity: MetricName, linkage: LinkageName,
                           max_distance: float) -> np.ndarray:
    """ Clusters rows from 'word_vectors' hierarchically with 'affinity' metric
    and 'linkage' criterion to compute linkage. Vectors within 'max_distance'
    form one cluster. Names the clusters according to 'token'. """
    if word_vectors.shape[0] < 2:
        return np.array([f"{token}_0"])

    sense_ids = AgglomerativeClustering(n_clusters=None,
                                        affinity=affinity.lower(),
                                        linkage=linkage.lower(),
                                        distance_threshold=max_distance) \
        .fit_predict(word_vectors)

    return np.array([f"{token}_{sense_id}" for sense_id in sense_ids])


def get_n_clusters_for_token(word_vectors: np.ndarray, token: str,
                             affinity: MetricName, linkage: LinkageName,
                             n_clusters: int) -> np.ndarray:
    """ Forms 'n_clusters' clusters of rows from 'word_vectors' hierarchically
    using the 'affinity' metric and 'linkage' criterion to compute linkage.
    Names the clusters according to 'token'. """
    if word_vectors.shape[0] < 2:
        return np.array([f"{token}_0"])

    sense_ids = AgglomerativeClustering(n_clusters=n_clusters,
                                        affinity=affinity.lower(),
                                        linkage=linkage.lower(),
                                        distance_threshold=None) \
        .fit_predict(word_vectors)

    return np.array([f"{token}_{sense_id}" for sense_id in sense_ids])


def cluster_vectors_per_token(word_vectors: np.ndarray,
                              id_map_reduced: pd.DataFrame,
                              affinity: MetricName, linkage: LinkageName,
                              max_distance: float) -> pd.DataFrame:
    """ Clusters rows of 'word_vectors' hierarchically per token using the
    'affinity' metric and 'linkage' criterion. Adds unique sense labels. Splits
    the dendrogram based on 'max_distance'. """
    id_map_reduced['sense'] = id_map_reduced.apply(
        lambda r: get_clusters_for_token(word_vectors[r.token_id],
                                         r.token, affinity, linkage,
                                         max_distance),
        axis=1)

    return id_map_reduced


def cluster_vectors_per_token_with_known_sense_count(
        word_vectors: np.ndarray, id_map_reduced: pd.DataFrame,
        affinity: MetricName, linkage: LinkageName) -> pd.DataFrame:
    """ Clusters rows of 'word_vectors' hierarchically per token from
    'id_map_reduced' using the 'affinity' metric and 'linkage' criterion. Adds
    unique sense labels. Requires the column 'unique_sense_count' in
    'id_map_reduced' with the number of clusters to find. """
    id_map_reduced['sense'] = id_map_reduced.apply(
        lambda r: get_n_clusters_for_token(word_vectors[r.token_id],
                                           r.token, affinity, linkage,
                                           r.unique_sense_count),
        axis=1)

    return id_map_reduced
