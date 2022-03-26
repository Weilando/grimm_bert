from unittest import TestCase, main
from unittest.mock import patch

import pandas as pd
from nltk.corpus import wordnet as wn
from nltk.tree.tree import Tree

from data import semcor_preprocessor as sp


class TestSemcorPreprocessor(TestCase):
    @classmethod
    def setUpClass(cls):
        """ Set up typical datastructures that can occur in a sentence. """
        cls.token_list = ['of']
        cls.token_tree_str = Tree('such.s.00', ['such'])
        cls.token_tree_lem = Tree(wn.lemma('state.v.01.say'), ['said'])
        cls.token_tree_sub = Tree(wn.lemma('group.n.01.group'),
                                  [Tree('NE', ['Jacob', 'Grimm'])])

    def test_get_tokens_and_senses_from_list(self):
        df_exp = pd.DataFrame({'token': ['of'], 'sense': ['of_S']})
        df_res = sp.get_tokens_and_senses_from_list(self.token_list, '_S')
        pd.testing.assert_frame_equal(df_exp, df_res)

    def test_get_tokens_and_senses_from_tree_with_str(self):
        df_exp = pd.DataFrame({'token': ['such'], 'sense': ['such_such.s.00']})
        df_res = sp.get_tokens_and_senses_from_tree(self.token_tree_str)
        pd.testing.assert_frame_equal(df_exp, df_res)

    def test_get_tokens_and_senses_from_tree_with_lemma(self):
        df_exp = pd.DataFrame({'token': ['said'],
                               'sense': ["said_Lemma('state.v.01.say')"]})
        df_res = sp.get_tokens_and_senses_from_tree(self.token_tree_lem)
        pd.testing.assert_frame_equal(df_exp, df_res)

    def test_get_tokens_and_senses_from_nested_tree(self):
        df_exp = pd.DataFrame({'token': ['Jacob', 'Grimm'],
                               'sense': ["Jacob_Lemma('group.n.01.group')",
                                         "Grimm_Lemma('group.n.01.group')"]})
        df_res = sp.get_tokens_and_senses_from_tree(self.token_tree_sub)
        pd.testing.assert_frame_equal(df_exp, df_res)

    def test_get_tokens_and_senses_from_sentence(self):
        sentence = [self.token_list, self.token_tree_lem, self.token_tree_str,
                    self.token_tree_sub]
        df_exp = pd.DataFrame({
            'token': ['of', 'said', 'such', 'Jacob', 'Grimm'],
            'sense': ['of_S', "said_Lemma('state.v.01.say')", 'such_such.s.00',
                      "Jacob_Lemma('group.n.01.group')",
                      "Grimm_Lemma('group.n.01.group')"]
        })
        df_res = sp.get_tokens_and_senses_from_sentence(sentence, '_S')
        pd.testing.assert_frame_equal(df_exp, df_res)

    def test_get_tokens_and_senses_from_sentences(self):
        sentences = [[self.token_list, self.token_tree_lem],
                     [self.token_tree_str, self.token_tree_sub]]
        df_exp = pd.DataFrame({
            'token': ['of', 'said', 'such', 'Jacob', 'Grimm'],
            'sense': ['of_S', "said_Lemma('state.v.01.say')", 'such_such.s.00',
                      "Jacob_Lemma('group.n.01.group')",
                      "Grimm_Lemma('group.n.01.group')"]
        })
        df_res = sp.get_tokens_and_senses_from_sentences(sentences, '_S')
        pd.testing.assert_frame_equal(df_exp, df_res)

    @patch('nltk.corpus.semcor')
    def test_get_tagged_sentences(self, mock_reader):
        mock_reader.tagged_sents.return_value = [3]
        self.assertEqual(sp.get_sentences_with_sense_tags(mock_reader), [3])
        mock_reader.tagged_sents.assert_called_with(tag="sem")


if __name__ == '__main__':
    main()
