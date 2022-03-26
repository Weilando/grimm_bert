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
        args = ['Toy', 'res', '-l', 'DEBUG', '-c', 'model_cache', '-d', '0.5']
        parsed_args = self.parser.parse_args(args)
        self.assertEqual(parsed_args.corpus_name, 'Toy')
        self.assertEqual(parsed_args.results_path, 'res')
        self.assertEqual(parsed_args.log, 'DEBUG')
        self.assertEqual(parsed_args.model_cache, 'model_cache')
        self.assertEqual(parsed_args.max_dist, 0.5)

    def test_parse_long_options(self):
        args = ['Toy', 'results_path', '--log', 'DEBUG', '--model_cache',
                'model_cache', '--max_dist', '0.5']
        parsed_args = self.parser.parse_args(args)
        self.assertEqual(parsed_args.corpus_name, 'Toy')
        self.assertEqual(parsed_args.results_path, 'results_path')
        self.assertEqual(parsed_args.log, 'DEBUG')
        self.assertEqual(parsed_args.model_cache, 'model_cache')
        self.assertEqual(parsed_args.max_dist, 0.5)

    def test_parse_defaults(self):
        args = ['Toy', 'results_path']
        parsed_args = self.parser.parse_args(args)
        self.assertEqual(parsed_args.corpus_name, 'Toy')
        self.assertEqual(parsed_args.results_path, 'results_path')
        self.assertEqual(parsed_args.log, gb.DEFAULT_LOG_LEVEL)
        self.assertEqual(parsed_args.model_cache, gb.DEFAULT_MODEL_CACHE_DIR)
        self.assertEqual(parsed_args.max_dist, gb.DEFAULT_MAX_CLUSTER_DISTANCE)


if __name__ == '__main__':
    main()
