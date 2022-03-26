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

    def test_get_hierarchical_cluster_single_class(self):
        distance_matrix = np.array([[0.]])
        cluster_exp = np.array([0])
        cluster = cl.get_hierarchical_cluster(distance_matrix, max_distance=2.)
        np.testing.assert_array_equal(cluster_exp, cluster)

    def test_get_hierarchical_cluster(self):
        distance_matrix = np.array([[0., 1., 7.], [1., 0., 7.], [7., 7., 0.]])
        cluster_exp = np.array([0, 0, 1])
        cluster = cl.get_hierarchical_cluster(distance_matrix, max_distance=2.)
        np.testing.assert_array_equal(cluster_exp, cluster)

    def test_cluster_vectors_per_token(self):
        word_vectors = np.array([[.9, .0], [.7, .1], [-.5, .5], [.8, .1]])
        id_map = pd.DataFrame({'token': ['b', 'a', 'b', 'b'],
                               'reference_id': [0, 0, 0, 1],
                               'word_vector_id': [0, 1, 2, 3]})
        dictionary_exp = pd.DataFrame({'token': ['b', 'a', 'b', 'b'],
                                       'reference_id': [0, 0, 0, 1],
                                       'word_vector_id': [0, 1, 2, 3],
                                       'sense': ['b_0', 'a_0', 'b_1', 'b_0']})

        dictionary_res = cl.cluster_vectors_per_token(word_vectors, id_map,
                                                      max_distance=0.5)
        pd.testing.assert_frame_equal(dictionary_exp, dictionary_res)

    def test_reduce_dictionary(self):
        dictionary = pd.DataFrame({'token': ['b', 'a', 'b', 'b'],
                                   'reference_id': [0, 0, 0, 1],
                                   'word_vector_id': [0, 1, 2, 3],
                                   'sense': ['b_0', 'a_0', 'b_1', 'b_0']})
        dictionary_exp = pd.DataFrame({
            'token': ['a', 'b', 'b'],
            'sense': ['a_0', 'b_0', 'b_1'],
            'reference_id': [[0], [0, 1], [0]],
            'word_vector_id': [[1], [0, 3], [2]]})
        dictionary_res = cl.reduce_dictionary(dictionary)
        pd.testing.assert_frame_equal(dictionary_exp, dictionary_res)

    def test_extract_int_senses(self):
        dictionary = pd.DataFrame({'sense': ['a0', 'b0', 'a1', 'a0', 'b1']})
        id_senses_exp = [0, 1, 2, 0, 3]
        id_senses_res = cl.extract_int_senses(dictionary)
        self.assertEqual(id_senses_exp, id_senses_res)


if __name__ == '__main__':
    main()
