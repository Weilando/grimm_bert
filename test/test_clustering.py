from unittest import main, TestCase

import numpy as np
from sklearn.cluster import AgglomerativeClustering

import analysis.clustering as ac


class TestClustering(TestCase):
    def test_get_hierarchical_cluster(self):
        distance_matrix = np.array([[0., 1., 7.], [1., 0., 7.], [7., 7., 0.]])
        expected_labels = np.array([0, 0, 1])

        cluster = ac.get_hierarchical_cluster(distance_matrix, max_distance=2.)

        self.assertIsInstance(cluster, AgglomerativeClustering)
        np.testing.assert_array_equal(expected_labels, cluster.labels_)


if __name__ == '__main__':
    main()
