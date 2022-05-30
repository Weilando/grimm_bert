import os
from tempfile import TemporaryFile, TemporaryDirectory
from unittest import TestCase, main
from xml.etree import ElementTree

import pandas as pd

import data.wsdeval_preprocessor as wp

TOY_XML_CORPUS = \
    """<?xml version="1.0" encoding="UTF-8" ?>
    <corpus lang="en" source="fake_semeval"><text id="d000">
    <sentence id="d000.s000">
    <wf lemma="this" pos="DET">This</wf>
    <instance id="d000.s000.t000" lemma="document" pos="NOUN">doc</instance>
    <wf lemma="be" pos="VERB">is</wf>
    <wf lemma="a" pos="DET">a</wf>
    <instance id="d000.s000.t001" lemma="summary" pos="NOUN">summary</instance>
    <wf lemma="." pos=".">.</wf>
    </sentence>
    <sentence id="d003.s011">
    <wf lemma="how" pos="ADV">How</wf>
    <wf lemma="do" pos="VERB">does</wf>
    <wf lemma="it" pos="PRON">it</wf>
    <wf lemma="work" pos="VERB">work</wf>
    <wf lemma="?" pos=".">?</wf>
    </sentence>
    </text></corpus>"""


class TestWsdevalPreprocessor(TestCase):
    @classmethod
    def setUp(cls):
        cls.xml_tree = ElementTree.fromstring(TOY_XML_CORPUS)
        cls.gold_keys = pd.DataFrame({
            'id': ['d000.s000.t000', 'd000.s000.t001'],
            'sense': ['document0', 'summary0']}).set_index('id')

    def test_get_xml_tree(self):
        with TemporaryFile(suffix='txt') as xml_file:
            xml_file.write(TOY_XML_CORPUS.encode("utf-8"))
            xml_file.seek(0)

            result_xml_tree = wp.get_xml_tree(xml_file)
            self.assertEqual(ElementTree.tostring(self.xml_tree),
                             ElementTree.tostring(result_xml_tree))

    def test_get_gold_keys(self):
        """ Should load ids and first occurring senses into a DataFrame. """
        with TemporaryFile(suffix='txt') as gold_keys_file:
            gold_keys_file.write(b"id0 sense0\nid1 sense1 sense2")
            gold_keys_file.seek(0)

            expected_gold_keys = pd.DataFrame({'id': ['id0', 'id1'],
                                               'sense': ['sense0', 'sense1']}) \
                .set_index('id')
            result_gold_keys = wp.get_gold_keys(gold_keys_file)
            pd.testing.assert_frame_equal(expected_gold_keys, result_gold_keys)

    def test_get_sentences(self):
        with TemporaryDirectory() as tmp_dir:
            os.chdir(tmp_dir)
            preprocessor = wp.WsdevalPreprocessor(self.xml_tree, self.gold_keys,
                                                  corpus_cache_path=tmp_dir)
            sentences = preprocessor.get_sentences()
        expected_sentences = pd.DataFrame({
            'sentence': [['this', 'doc', 'is', 'a', 'summary', '.'],
                         ['how', 'does', 'it', 'work', '?']]})
        pd.testing.assert_frame_equal(expected_sentences, sentences)

    def test_get_tagged_tokens(self):
        with TemporaryDirectory() as tmp_dir:
            os.chdir(tmp_dir)
            preprocessor = wp.WsdevalPreprocessor(self.xml_tree, self.gold_keys,
                                                  corpus_cache_path=tmp_dir)
            tokens = preprocessor.get_tagged_tokens()
        expected_tokens = pd.DataFrame({
            'token': ['this', 'doc', 'is', 'a', 'summary', '.', 'how', 'does',
                      'it', 'work', '?'],
            'sense': ['this_SENSE', 'doc_document0', 'is_SENSE', 'a_SENSE',
                      'summary_summary0', '._SENSE', 'how_SENSE', 'does_SENSE',
                      'it_SENSE', 'work_SENSE', '?_SENSE'],
            'tagged_sense': [False, True, False, False, True, False, False,
                             False, False, False, False]})
        pd.testing.assert_frame_equal(expected_tokens, tokens)


if __name__ == '__main__':
    main()
