from unittest import main, TestCase

import numpy as np
import pandas as pd

from analysis import clustering as cl


class TestClustering(TestCase):
    def test_is_square_matrix(self):
        matrix = np.array([[1, 2], [3, 4]])
        self.assertTrue(cl.is_square_matrix(matrix))

    def test_is_square_matrix_wrong_shapes(self):
        vector = np.array([1, 2])
        matrix = np.array([[1, 2], [3, 4], [5, 6]])
        tensor = np.array([[[1, 2], [3, 4]]])
        self.assertFalse(cl.is_square_matrix(vector))
        self.assertFalse(cl.is_square_matrix(matrix))
        self.assertFalse(cl.is_square_matrix(tensor))

    def test_extract_sub_matrix(self):
        distance_matrix = np.arange(start=1, stop=10).reshape(3, 3)
        indices = [2]
        expected_matrix = np.array([[9]])
        result_matrix = cl.extract_sub_matrix(distance_matrix, indices)
        np.testing.assert_array_equal(expected_matrix, result_matrix)

    def test_get_hierarchical_cluster_single_class(self):
        distance_matrix = np.array([[0.]])
        expected_labels = np.array([0])
        cluster = cl.get_hierarchical_cluster(distance_matrix, max_distance=2.)
        np.testing.assert_array_equal(expected_labels, cluster)

    def test_get_hierarchical_cluster(self):
        distance_matrix = np.array([[0., 1., 7.], [1., 0., 7.], [7., 7., 0.]])
        expected_labels = np.array([0, 0, 1])
        cluster = cl.get_hierarchical_cluster(distance_matrix, max_distance=2.)
        np.testing.assert_array_equal(expected_labels, cluster)

    def test_cluster_vectors_per_token(self):
        distance_matrix = np.array([[.0, .1, .6, .1],
                                    [.1, .0, .7, .8],
                                    [.6, .7, .0, .8],
                                    [.1, .8, .8, .0]])
        id_map = pd.DataFrame({'token': ['b', 'a', 'b', 'b'],
                               'reference_id': [0, 0, 0, 1],
                               'word_vector_id': [0, 1, 2, 3]})
        id_map_reduced = pd.DataFrame({'token': ['b', 'a'],
                                       'reference_id': [[0, 1], [0]],
                                       'word_vector_id': [[0, 2, 3], [1]]})
        expected_dictionary = pd.DataFrame({
            'token': ['a', 'b', 'b'],
            'sense': ['a_0', 'b_0', 'b_1'],
            'reference_id': [[0], [0, 1], [0]],
            'word_vector_id': [[1], [0, 3], [2]]})
        result_dictionary = cl.cluster_vectors_per_token(
            distance_matrix, id_map, id_map_reduced, max_distance=0.5)
        pd.testing.assert_frame_equal(expected_dictionary, result_dictionary)


if __name__ == '__main__':
    main()
