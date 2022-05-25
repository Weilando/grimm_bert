from typing import Tuple, cast

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from clustering.linkage_name import LinkageName
from clustering.metric_name import MetricName


def get_clusters_for_token_via_cluster_count(
        word_vectors: np.ndarray, token: str, affinity: MetricName,
        linkage: LinkageName, cluster_count: int) \
        -> Tuple[np.ndarray, np.ndarray]:
    """ Forms 'cluster_count' clusters of rows from 'word_vectors'
    hierarchically using the 'affinity' metric and 'linkage' criterion to
    compute linkage. Names the clusters according to 'token'. Also returns the
    last linkage distance before and after the cut if they exist. """
    if word_vectors.shape[0] < 2:
        return np.array([f"{token}_0"]), np.array([], dtype=float)

    clustering = AgglomerativeClustering(n_clusters=cluster_count,
                                         affinity=affinity.lower(),
                                         linkage=linkage.lower(),
                                         distance_threshold=None,
                                         compute_distances=True) \
        .fit(word_vectors)
    clustering = cast(AgglomerativeClustering, clustering)

    sense_labels = np.array([f"{token}_{sense_id}" for sense_id
                             in clustering.labels_])

    if cluster_count == 2:
        last_cut_distances = clustering.distances_[-cluster_count:]
    else:
        last_cut_distances = clustering.distances_[-cluster_count:
                                                   -cluster_count + 2]

    return sense_labels, last_cut_distances


def get_clusters_for_token_via_max_distance(
        word_vectors: np.ndarray, token: str, affinity: MetricName,
        linkage: LinkageName, max_distance: float) -> np.ndarray:
    """ Clusters rows from 'word_vectors' hierarchically using 'affinity' metric
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


def get_clusters_for_token_via_silhouette(
        word_vectors: np.ndarray, token: str, affinity: MetricName,
        linkage: LinkageName, min_silhouette: float) -> np.ndarray:
    """ Clusters rows from 'word_vectors' hierarchically with 'affinity' metric
    and 'linkage' criterion. Increases the sense count iteratively and keeps the
    one with the highest Silhouette score. Presumes 'min_silhouette' for a
    single sense. Names the clusters according to 'token'. """
    if word_vectors.shape[0] < 2:
        return np.array([f"{token}_0"])
    token_count = word_vectors.shape[0]

    best_silhouette = min_silhouette
    best_clustering = np.zeros(token_count, dtype=int)
    max_sense_count = np.floor(np.sqrt(token_count)).astype(int)

    for sense_count in np.arange(start=2, stop=max_sense_count + 1):
        sense_ids = AgglomerativeClustering(
            n_clusters=sense_count, affinity=affinity.lower(),
            linkage=linkage.lower(), distance_threshold=None) \
            .fit_predict(word_vectors)
        current_silhouette = silhouette_score(word_vectors, sense_ids)
        if current_silhouette > best_silhouette:
            best_silhouette = current_silhouette
            best_clustering = sense_ids

    return np.array([f"{token}_{sense_id}" for sense_id in best_clustering])


def cluster_vectors_per_token_with_known_sense_count(
        word_vectors: np.ndarray, id_map_reduced: pd.DataFrame,
        affinity: MetricName, linkage: LinkageName) -> pd.DataFrame:
    """ Clusters rows of 'word_vectors' hierarchically per token from
    'id_map_reduced' using the 'affinity' metric and 'linkage' criterion. Adds
    unique sense labels. Requires the column 'unique_sense_count' in
    'id_map_reduced' with the number of clusters to find. """
    id_map_reduced[['sense', 'last_cut_distances']] = id_map_reduced.apply(
        lambda r: get_clusters_for_token_via_cluster_count(
            word_vectors[r.token_id], r.token, affinity, linkage,
            r.unique_sense_count),
        axis=1, result_type='expand')

    return id_map_reduced


def cluster_vectors_per_token_with_max_distance(
        word_vectors: np.ndarray, id_map_reduced: pd.DataFrame,
        affinity: MetricName, linkage: LinkageName, max_distance: float) \
        -> pd.DataFrame:
    """ Clusters rows of 'word_vectors' hierarchically per token using the
    'affinity' metric and 'linkage' criterion. Adds unique sense labels. Splits
    the dendrogram based on 'max_distance'. """
    id_map_reduced['sense'] = id_map_reduced.apply(
        lambda r: get_clusters_for_token_via_max_distance(
            word_vectors[r.token_id], r.token, affinity, linkage, max_distance),
        axis=1)

    return id_map_reduced


def cluster_vectors_per_token_with_silhouette_criterion(
        word_vectors: np.ndarray, id_map_reduced: pd.DataFrame,
        affinity: MetricName, linkage: LinkageName, min_silhouette: float) \
        -> pd.DataFrame:
    """ Clusters rows of 'word_vectors' hierarchically per token using the
    'affinity' metric and 'linkage' criterion. Adds unique sense labels.
    Increases the sense count iteratively and keeps the one with the highest
    Silhouette score. Presumes 'min_silhouette' for a single sense. """
    id_map_reduced['sense'] = id_map_reduced.apply(
        lambda r: get_clusters_for_token_via_silhouette(
            word_vectors[r.token_id], r.token, affinity, linkage,
            min_silhouette),
        axis=1)

    return id_map_reduced
