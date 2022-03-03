from unittest import main, TestCase

import numpy as np
import pandas as pd

import analysis.clustering as ac


class TestClustering(TestCase):
    def test_is_square_matrix(self):
        matrix = np.array([[1, 2], [3, 4]])
        self.assertTrue(ac.is_square_matrix(matrix))

    def test_is_square_matrix_wrong_shapes(self):
        vector = np.array([1, 2])
        matrix = np.array([[1, 2], [3, 4], [5, 6]])
        tensor = np.array([[[1, 2], [3, 4]]])
        self.assertFalse(ac.is_square_matrix(vector))
        self.assertFalse(ac.is_square_matrix(matrix))
        self.assertFalse(ac.is_square_matrix(tensor))

    def test_extract_sub_matrix(self):
        distance_matrix = np.arange(start=1, stop=10).reshape(3, 3)
        indices = [2]
        expected_matrix = np.array([[9]])
        result_matrix = ac.extract_sub_matrix(distance_matrix, indices)
        np.testing.assert_array_equal(expected_matrix, result_matrix)

    def test_get_hierarchical_cluster_single_class(self):
        distance_matrix = np.array([[0.]])
        expected_labels = np.array([0])
        cluster = ac.get_hierarchical_cluster(distance_matrix, max_distance=2.)
        np.testing.assert_array_equal(expected_labels, cluster)

    def test_get_hierarchical_cluster(self):
        distance_matrix = np.array([[0., 1., 7.], [1., 0., 7.], [7., 7., 0.]])
        expected_labels = np.array([0, 0, 1])
        cluster = ac.get_hierarchical_cluster(distance_matrix, max_distance=2.)
        np.testing.assert_array_equal(expected_labels, cluster)

    def test_cluster_vectors_per_token(self):
        distance_matrix = np.array([[0., 1., 3.], [1., 0., 2.], [3., 2., 0.]])
        id_map = pd.DataFrame({'token': [0, 3, 0],
                               'reference_id': [0, 0, 1],
                               'word_vector_id': [0, 1, 2]})
        id_map_reduced = pd.DataFrame({'token': [0, 3],
                                       'reference_id': [[0, 1], [0]],
                                       'word_vector_id': [[0, 2], [1]]})
        expected_dictionary = pd.DataFrame({'token': [0, 0, 3],
                                            'meaning_label': [0, 1, 0],
                                            'reference_id': [[1], [0], [0]],
                                            'word_vector_id': [[2], [0], [1]]})
        result_dictionary = ac.cluster_vectors_per_token(distance_matrix,
                                                         id_map, id_map_reduced,
                                                         max_distance=0.5)
        pd.testing.assert_frame_equal(expected_dictionary, result_dictionary)


if __name__ == '__main__':
    main()
