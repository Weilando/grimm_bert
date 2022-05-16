from unittest import main, TestCase

import numpy as np
import pandas as pd

import clustering.hierarchical_clustering as hc
from clustering.linkage_name import LinkageName
from clustering.metric_name import MetricName


class TestNameDicts(TestCase):
    def test_get_metric_name_values(self):
        self.assertEqual(['Cosine', 'Euclidean'], MetricName.get_values())

    def test_get_linkage_name_values(self):
        self.assertEqual(['Average', 'Complete', 'Single'],
                         LinkageName.get_values())


class TestClustering(TestCase):
    def test_get_clusters_for_token_via_cluster_count_2_senses(self):
        """ Should generate two clusters and assign the last word vector to a
        single cluster, as its cosine distance to the other vectors is high. """
        word_vectors = np.array([[.9, .0], [.7, .1], [-.5, -.1]])
        cluster_exp = np.array(['t_0', 't_0', 't_1'])
        cluster = hc.get_clusters_for_token_via_cluster_count(
            word_vectors, 't', MetricName.COSINE, LinkageName.SINGLE, 2)
        np.testing.assert_array_equal(cluster_exp, cluster)

    def test_get_clusters_for_token_via_cluster_count_3_senses(self):
        """ Should generate three clusters and assign each word vector to its
        own cluster. """
        word_vectors = np.array([[.9, .0], [.7, .1], [-.5, -.1]])
        cluster_exp = np.array(['t_2', 't_1', 't_0'])
        cluster = hc.get_clusters_for_token_via_cluster_count(
            word_vectors, 't', MetricName.COSINE, LinkageName.SINGLE, 3)
        np.testing.assert_array_equal(cluster_exp, cluster)

    def test_get_clusters_for_token_via_max_distance_1_token(self):
        """ Should assign one word vector to one cluster. """
        word_vectors = np.array([[.9, .0]])
        cluster_exp = np.array(['t_0'])
        cluster = hc.get_clusters_for_token_via_max_distance(
            word_vectors, 't', MetricName.COSINE, LinkageName.SINGLE, 0.5)
        np.testing.assert_array_equal(cluster_exp, cluster)

    def test_get_clusters_for_token_via_max_distance_3_tokens(self):
        """ Should assign the last word vector to a different cluster, as its
        cosine distance to the other vectors is high. """
        word_vectors = np.array([[.9, .0], [.7, .1], [-.5, -.1]])
        cluster_exp = np.array(['t_0', 't_0', 't_1'])
        cluster = hc.get_clusters_for_token_via_max_distance(
            word_vectors, 't', MetricName.COSINE, LinkageName.SINGLE, 0.5)
        np.testing.assert_array_equal(cluster_exp, cluster)

    def test_get_clusters_for_token_via_silhouette_2_tokens(self):
        """ Should assign both vectors to the same cluster. """
        word_vectors = np.array([[.9, .0], [.7, .1]])
        cluster_exp = np.array(['t_0', 't_0'])
        cluster = hc.get_clusters_for_token_via_silhouette(
            word_vectors, 't', MetricName.COSINE, LinkageName.SINGLE, 0.1)
        np.testing.assert_array_equal(cluster_exp, cluster)

    def test_get_clusters_for_token_via_silhouette_4_tokens(self):
        """ Should assign the negative vector to its own cluster. """
        word_vectors = np.array([[.9, .0], [.7, .1], [-.5, -.1], [.8, .1]])
        cluster_exp = np.array(['t_0', 't_0', 't_1', 't_0'])
        cluster = hc.get_clusters_for_token_via_silhouette(
            word_vectors, 't', MetricName.COSINE, LinkageName.SINGLE, 0.1)
        np.testing.assert_array_equal(cluster_exp, cluster)

    def test_cluster_vectors_per_token_with_known_sense_count(self):
        """ Should assign the correct clusters per token. """
        word_vectors = np.array([[.9, .0], [.7, .1], [-.5, .5], [.8, .1]])
        id_map_red = pd.DataFrame({'token': ['a', 'b'],
                                   'sentence_id': [[0], [0, 0, 1]],
                                   'token_id': [[1], [0, 2, 3]],
                                   'unique_sense_count': [1, 2]})
        dictionary_exp = pd.DataFrame({'token': ['a', 'b'],
                                       'sentence_id': [[0], [0, 0, 1]],
                                       'token_id': [[1], [0, 2, 3]],
                                       'unique_sense_count': [1, 2],
                                       'sense': [['a_0'], ['b_0', 'b_1', 'b_0']]
                                       })

        dictionary_res = hc.cluster_vectors_per_token_with_known_sense_count(
            word_vectors, id_map_red, MetricName.COSINE, LinkageName.SINGLE)
        pd.testing.assert_frame_equal(dictionary_exp, dictionary_res)

    def test_cluster_vectors_per_token_with_max_distance(self):
        """ Should assign the correct clusters per token. """
        word_vectors = np.array([[.9, .0], [.7, .1], [-.5, .5], [.8, .1]])
        id_map_red = pd.DataFrame({'token': ['a', 'b'],
                                   'sentence_id': [[0], [0, 0, 1]],
                                   'token_id': [[1], [0, 2, 3]]})
        dictionary_exp = pd.DataFrame({'token': ['a', 'b'],
                                       'sentence_id': [[0], [0, 0, 1]],
                                       'token_id': [[1], [0, 2, 3]],
                                       'sense': [['a_0'], ['b_0', 'b_1', 'b_0']]
                                       })

        dictionary_res = hc.cluster_vectors_per_token_with_max_distance(
            word_vectors, id_map_red, MetricName.COSINE, LinkageName.SINGLE, .5)
        pd.testing.assert_frame_equal(dictionary_exp, dictionary_res)

    def test_cluster_vectors_per_token_with_silhouette_criterion(self):
        """ Should assign the correct clusters per token. """
        word_vectors = np.array([[.9, .0], [.7, .1], [-.5, .5], [.8, .1]])
        id_map_red = pd.DataFrame({'token': ['a', 'b'],
                                   'sentence_id': [[0], [0, 0, 1]],
                                   'token_id': [[1], [0, 2, 3]]})
        dictionary_exp = pd.DataFrame({'token': ['a', 'b'],
                                       'sentence_id': [[0], [0, 0, 1]],
                                       'token_id': [[1], [0, 2, 3]],
                                       'sense': [['a_0'], ['b_0', 'b_0', 'b_0']]
                                       })

        dictionary_res = hc.cluster_vectors_per_token_with_silhouette_criterion(
            word_vectors, id_map_red, MetricName.COSINE, LinkageName.SINGLE, .1)
        pd.testing.assert_frame_equal(dictionary_exp, dictionary_res)


if __name__ == '__main__':
    main()
