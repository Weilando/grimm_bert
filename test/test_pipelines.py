import os
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import Mock

import pandas as pd

import aggregation.pipelines
from clustering.linkage_name import LinkageName
from clustering.metric_name import MetricName
from data.corpus_handler import CorpusHandler
from data.corpus_name import CorpusName
from grimm_bert import DEFAULT_MODEL_CACHE_PATH


class TestPipelines(TestCase):
    def setUp(self):
        self.config = {'corpus_name': CorpusName.TOY,
                       'get_sentences.return_value': pd.DataFrame({
                           'sentence': [['hello', 'world', '!']]}),
                       'get_sentences_as_list.return_value':
                           [['hello', 'world', '!']],
                       'get_tagged_tokens.return_value': pd.DataFrame({
                           'token': ['hello', 'world', '!'],
                           'sense': ['a', 'b', 'c'],
                           'tagged_sense': [True, True, False]})
                       }
        current_path = os.path.dirname(os.path.abspath(__file__))
        os.chdir(os.path.join(current_path, '..'))

    def test_create_dictionary_with_known_sense_counts(self):
        """ Should execute the pipeline without errors. """
        with TemporaryDirectory() as res_path:
            aggregation.pipelines.create_dictionary_with_known_sense_counts(
                Mock(CorpusHandler, **self.config), DEFAULT_MODEL_CACHE_PATH,
                res_path, MetricName.EUCLIDEAN, LinkageName.SINGLE, 'exp_name',
                {'a': 1})

    def test_create_dictionary_with_max_distance(self):
        """ Should execute the pipeline without errors. """
        with TemporaryDirectory() as res_path:
            aggregation.pipelines.create_dictionary_with_max_distance(
                Mock(CorpusHandler, **self.config),
                DEFAULT_MODEL_CACHE_PATH, res_path,
                MetricName.EUCLIDEAN, LinkageName.SINGLE, 0.2, 'exp_name',
                {'a': 1})

    def test_create_dictionary_with_min_silhouette(self):
        """ Should execute the pipeline without errors. """
        with TemporaryDirectory() as res_path:
            aggregation.pipelines.create_dictionary_with_min_silhouette(
                Mock(CorpusHandler, **self.config),
                DEFAULT_MODEL_CACHE_PATH, res_path,
                MetricName.EUCLIDEAN, LinkageName.SINGLE, 0.1, 'exp_name',
                {'a': 1})


if __name__ == '__main__':
    main()
