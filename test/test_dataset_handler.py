from unittest import TestCase, main, mock

import pandas as pd
from nltk.corpus.reader.wordnet import Lemma
from nltk.tree.tree import Tree

import data.dataset_handler as dh


class TestDatasetHandler(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.token_list = ['list_token']
        cls.token_tree_str = Tree('tree_sense_str', ['tree_token_str'])
        cls.token_tree_lem = Tree(Lemma(wordnet_corpus_reader=None,
                                        synset=None,
                                        name='tree_sense_lem',
                                        lexname_index=None,
                                        lex_id=None,
                                        syntactic_marker=None),
                                  ['tree_token'])
        cls.nested_token_tree = Tree(Lemma(wordnet_corpus_reader=None,
                                           synset=None,
                                           name='nested_tree_sense',
                                           lexname_index=None,
                                           lex_id=None,
                                           syntactic_marker=None),
                                     [Tree('NE', ['Jacob', 'Grimm'])])

    def test_extract_tokens_and_senses_from_list(self):
        expected_df = pd.DataFrame(data={'token': self.token_list,
                                         'sense': [dh.STD_SENSE]})
        result_df = dh.extract_tokens_and_senses_from_list(self.token_list)
        pd.testing.assert_frame_equal(expected_df, result_df)

    def test_extract_tokens_and_senses_from_tree_with_str(self):
        expected_df = pd.DataFrame(data={'token': ['tree_token_str'],
                                         'sense': ['tree_sense_str']})
        result_df = dh.extract_tokens_and_senses_from_tree(self.token_tree_str)
        pd.testing.assert_frame_equal(expected_df, result_df)

    def test_extract_tokens_and_senses_from_tree_with_lemma(self):
        expected_df = pd.DataFrame(data={'token': ['tree_token'],
                                         'sense': ['tree_sense_lem']})
        result_df = dh.extract_tokens_and_senses_from_tree(self.token_tree_lem)
        pd.testing.assert_frame_equal(expected_df, result_df)

    def test_extract_tokens_and_senses_from_nested_tree(self):
        expected_df = pd.DataFrame(data={'token': ['Jacob', 'Grimm'],
                                         'sense': ['nested_tree_sense',
                                                   'nested_tree_sense']})
        result_df = dh.extract_tokens_and_senses_from_tree(
            self.nested_token_tree)
        pd.testing.assert_frame_equal(expected_df, result_df)

    def test_extract_tokens_and_senses_from_sentence(self):
        sentence = [self.token_list, self.token_tree_lem, self.token_tree_str,
                    self.nested_token_tree]
        expected_df = pd.DataFrame(data={'token': ['list_token', 'tree_token',
                                                   'tree_token_str', 'Jacob',
                                                   'Grimm'],
                                         'sense': [dh.STD_SENSE,
                                                   'tree_sense_lem',
                                                   'tree_sense_str',
                                                   'nested_tree_sense',
                                                   'nested_tree_sense']})
        result_df = dh.extract_tokens_and_senses_from_sentence(sentence)
        pd.testing.assert_frame_equal(expected_df, result_df)

    def test_get_tagged_sentences(self):
        with mock.patch('nltk.corpus.semcor') as mock_reader:
            mock_reader.tagged_sents.return_value = [42]
            self.assertEqual(dh.get_tagged_sentences(mock_reader), [42])
            mock_reader.tagged_sents.assert_called_with(tag="sem")


if __name__ == '__main__':
    main()
