from argparse import ArgumentParser
from unittest import TestCase, main

import grimm_bert as gb


class TestGrimmBert(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = gb.build_argument_parser()

    def test_build_argument_parser(self):
        self.assertIsInstance(self.parser, ArgumentParser)

    def test_parse_short_options(self):
        args = ['Toy', '-r', 'R', '-l', 'L', '-m', 'M', '-c', 'C', '-d', '.5']
        parsed_args = self.parser.parse_args(args)
        self.assertEqual(parsed_args.corpus_name, 'Toy')
        self.assertEqual(parsed_args.corpus_cache, 'C')
        self.assertEqual(parsed_args.model_cache, 'M')
        self.assertEqual(parsed_args.results_path, 'R')
        self.assertEqual(parsed_args.max_dist, 0.5)
        self.assertEqual(parsed_args.log, 'L')

    def test_parse_long_options(self):
        args = ['Toy', '--results_path', 'rp', '--log', 'INFO', '--model_cache',
                'md', '--corpus_cache', 'cd', '--max_dist', '0.5']
        parsed_args = self.parser.parse_args(args)
        self.assertEqual(parsed_args.corpus_name, 'Toy')
        self.assertEqual(parsed_args.corpus_cache, 'cd')
        self.assertEqual(parsed_args.model_cache, 'md')
        self.assertEqual(parsed_args.results_path, 'rp')
        self.assertEqual(parsed_args.max_dist, 0.5)
        self.assertEqual(parsed_args.log, 'INFO')

    def test_parse_defaults(self):
        args = ['Toy']
        parsed_args = self.parser.parse_args(args)
        self.assertEqual(parsed_args.corpus_name, 'Toy')
        self.assertEqual(parsed_args.corpus_cache, gb.DEFAULT_CORPUS_CACHE_DIR)
        self.assertEqual(parsed_args.model_cache, gb.DEFAULT_MODEL_CACHE_DIR)
        self.assertEqual(parsed_args.results_path, gb.DEFAULT_RESULTS_PATH)
        self.assertEqual(parsed_args.max_dist, gb.DEFAULT_MAX_CLUSTER_DISTANCE)
        self.assertEqual(parsed_args.log, gb.DEFAULT_LOG_LEVEL)


if __name__ == '__main__':
    main()
