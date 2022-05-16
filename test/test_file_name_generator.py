from unittest import main, TestCase

import data.file_name_generator as fg
from data.corpus_name import CorpusName


class TestFileName(TestCase):
    def test_gen_dictionary_file_name(self):
        self.assertEqual('ep-dictionary.pkl', fg.gen_dictionary_file_name('ep'))

    def test_gen_raw_id_map_file_name(self):
        self.assertEqual('toy-raw_id_map.pkl',
                         fg.gen_raw_id_map_file_name(CorpusName.TOY))

    def test_gen_sentences_file_name(self):
        self.assertEqual('toy-sentences.pkl',
                         fg.gen_sentences_file_name(CorpusName.TOY))

    def test_gen_stats_file_name(self):
        self.assertEqual('ep-stats.json', fg.gen_stats_file_name('ep'))

    def test_gen_tagged_tokens_file_name(self):
        self.assertEqual('toy-tagged_tokens.pkl',
                         fg.gen_tagged_tokens_file_name(CorpusName.TOY))

    def test_gen_word_vec_file_name(self):
        self.assertEqual('toy-word_vectors.npz',
                         fg.gen_word_vec_file_name(CorpusName.TOY))


if __name__ == '__main__':
    main()
