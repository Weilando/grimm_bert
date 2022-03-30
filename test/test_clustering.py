from unittest import main, TestCase

import numpy as np
import pandas as pd

from analysis import clustering as cl


class TestClustering(TestCase):
    def test_get_hierarchical_cluster_single_class(self):
        """ Should assign one word vector to one cluster. """
        word_vectors = np.array([[.9, .0]])
        cluster_exp = np.array(['t_0'])
        cluster = cl.get_hierarchical_cluster(word_vectors, 't', .5)
        np.testing.assert_array_equal(cluster_exp, cluster)

    def test_get_hierarchical_cluster(self):
        """ Should assign the last word vector to a different cluster, as its
        cosine distance to the other vectors is high. """
        word_vectors = np.array([[.9, .0], [.7, .1], [-.5, -.1]])
        cluster_exp = np.array(['t_0', 't_0', 't_1'])
        cluster = cl.get_hierarchical_cluster(word_vectors, 't', .5)
        np.testing.assert_array_equal(cluster_exp, cluster)

    def test_cluster_vectors_per_token(self):
        """ Should assign the correct clusters per token. """
        word_vectors = np.array([[.9, .0], [.7, .1], [-.5, .5], [.8, .1]])
        id_map_red = pd.DataFrame({'token': ['a', 'b'],
                                   'reference_id': [[0], [0, 0, 1]],
                                   'word_vector_id': [[1], [0, 2, 3]]})
        dictionary_exp = pd.DataFrame({'token': ['a', 'b'],
                                       'reference_id': [[0], [0, 0, 1]],
                                       'word_vector_id': [[1], [0, 2, 3]],
                                       'sense': [['a_0'], ['b_0', 'b_1', 'b_0']]
                                       })

        dictionary_res = cl.cluster_vectors_per_token(word_vectors, id_map_red,
                                                      max_distance=0.5)
        pd.testing.assert_frame_equal(dictionary_exp, dictionary_res)

    def test_extract_flat_senses(self):
        """ Should unpack and sort senses regarding their word_vector_ids.
        Should drop reference_ids. """
        dictionary = pd.DataFrame({'word_vector_id': [[1, 2], [0]],
                                   'reference_ids': [[0, 1], [1]],
                                   'sense': [['c', 'a'], ['b']]})
        dictionary_exp = pd.DataFrame({'word_vector_id': [0, 1, 2],
                                       'sense': ['b', 'c', 'a']}) \
            .set_index('word_vector_id')
        dictionary_res = cl.extract_flat_senses(dictionary)
        pd.testing.assert_frame_equal(dictionary_exp, dictionary_res)

    def test_extract_int_senses(self):
        """ Should generate a unique integer labels per sense. """
        dictionary = pd.DataFrame({'sense': ['a', 'b', 'c', 'a', 'd']})
        id_senses_exp = [0, 1, 2, 0, 3]
        id_senses_res = cl.extract_int_senses(dictionary)
        self.assertEqual(id_senses_exp, id_senses_res)


if __name__ == '__main__':
    main()
