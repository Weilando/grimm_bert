from xml.etree import ElementTree

import pandas as pd

from data.corpus_handler import CorpusName
from data.corpus_preprocessor import CorpusPreprocessor
from grimm_bert import DEFAULT_CORPUS_CACHE_DIR

STD_SENSE = '_SENSE'


def get_xml_tree(xml_file_path: str) -> ElementTree.Element:
    """ Parses the XML file into an ElementTree. """
    tree = ElementTree.parse(xml_file_path)
    return tree.getroot()


def get_gold_keys(gold_key_file_path: str) -> pd.DataFrame:
    """ Parses the text file into a DataFrame with ids and senses. Chooses the
    first sense if multiple exist. """
    return pd.read_csv(gold_key_file_path, sep=" ", header=None,
                       names=["id", "sense"], index_col="id")


def add_senses_and_simplify_xml(xml_tree: ElementTree.Element,
                                gold_keys: pd.DataFrame) -> ElementTree.Element:
    """ Adds sense tags to 'xml_tree' and transforms 'wf' and 'instance' tags
    into 'token' tags. Generates sense names with the token (lemma) and a sense
    suffix. The sense is either the standard sense for 'wf', or the gold key
    sense for 'instance'. """
    for sentence in xml_tree.iter('sentence'):
        for token in sentence.iter('wf'):
            token.tag = 'token'
            token.set('sense', f"{token.text.lower()}{STD_SENSE}")
        for token in sentence.iter('instance'):
            token.tag = 'token'
            sense = gold_keys.loc[token.get('id')].sense
            token.set('sense', f"{token.text.lower()}_{sense}")
    return xml_tree


class WsdevalPreprocessor(CorpusPreprocessor):
    def __init__(self, xml_tree: ElementTree.Element, gold_keys: pd.DataFrame,
                 corpus_name: CorpusName = CorpusName.SEMEVAL15,
                 corpus_cache_path: str = DEFAULT_CORPUS_CACHE_DIR):
        super().__init__(corpus_name, corpus_cache_path)
        assert corpus_name.is_wsdeval_name
        self.xml_tree = add_senses_and_simplify_xml(xml_tree, gold_keys)

    def get_sentences(self) -> pd.DataFrame:
        sentences = [[token.text.lower() for token in sentence.iter('token')]
                     for sentence in self.xml_tree.iter('sentence')]
        return pd.DataFrame({'sentence': sentences})

    def get_tagged_tokens(self) -> pd.DataFrame:
        tokens = [token.text.lower() for token in self.xml_tree.iter('token')]
        senses = [token.get('sense') for token in self.xml_tree.iter('token')]
        return pd.DataFrame({'token': tokens, 'sense': senses})


if __name__ == '__main__':
    corpora = [CorpusName.SEMEVAL07, CorpusName.SEMEVAL13, CorpusName.SEMEVAL15,
               CorpusName.SENSEVAL2, CorpusName.SENSEVAL3, CorpusName.SEMCOR]

    for corpus in corpora:
        wsd_eval_preprocessor = WsdevalPreprocessor(
            get_xml_tree(f'data/wsdeval_corpora/{corpus}.data.xml'),
            get_gold_keys(f'data/wsdeval_corpora/{corpus}.gold.key.txt'),
            corpus)
        wsd_eval_preprocessor.cache_dataset()
