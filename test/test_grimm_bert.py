from argparse import ArgumentParser, ArgumentError
from io import StringIO
from unittest import TestCase, main
from unittest.mock import patch

import grimm_bert as gb


class TestGrimmBert(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = gb.build_argument_parser()

    def test_build_argument_parser(self):
        self.assertIsInstance(self.parser, ArgumentParser)

    def test_parse_short_options(self):
        args = ['Toy', 'average', '-r', 'R', '-l', 'L', '-m', 'M', '-c', 'C',
                '-d', '.5']
        parsed_args = self.parser.parse_args(args)
        self.assertEqual(parsed_args.corpus_name, 'Toy')
        self.assertEqual(parsed_args.linkage_name, 'average')
        self.assertEqual(parsed_args.corpus_cache, 'C')
        self.assertEqual(parsed_args.model_cache, 'M')
        self.assertEqual(parsed_args.results_path, 'R')
        self.assertEqual(parsed_args.max_dist, 0.5)
        self.assertEqual(parsed_args.log, 'L')

    def test_parse_long_options(self):
        args = ['Toy', 'complete', '--results_path', 'rp', '--log', 'INFO',
                '--model_cache', 'md', '--corpus_cache', 'cd',
                '--max_dist', '0.5']
        parsed_args = self.parser.parse_args(args)
        self.assertEqual(parsed_args.corpus_name, 'Toy')
        self.assertEqual(parsed_args.linkage_name, 'complete')
        self.assertEqual(parsed_args.corpus_cache, 'cd')
        self.assertEqual(parsed_args.model_cache, 'md')
        self.assertEqual(parsed_args.results_path, 'rp')
        self.assertEqual(parsed_args.max_dist, 0.5)
        self.assertEqual(parsed_args.log, 'INFO')

    def test_parse_defaults(self):
        args = ['Toy', 'single']
        parsed_args = self.parser.parse_args(args)
        self.assertEqual(parsed_args.corpus_name, 'Toy')
        self.assertEqual(parsed_args.linkage_name, 'single')
        self.assertEqual(parsed_args.corpus_cache, gb.DEFAULT_CORPUS_CACHE_DIR)
        self.assertEqual(parsed_args.model_cache, gb.DEFAULT_MODEL_CACHE_DIR)
        self.assertEqual(parsed_args.results_path, gb.DEFAULT_RESULTS_PATH)
        self.assertIsNone(parsed_args.max_dist)
        self.assertEqual(parsed_args.log, gb.DEFAULT_LOG_LEVEL)

    @patch('sys.stderr', new_callable=StringIO)
    def test_parse_no_max_dist(self, mock_stderr):
        """ Should raise an ArgumentError on empty max_dist argument. """
        with self.assertRaises(ArgumentError) and self.assertRaises(SystemExit):
            self.parser.parse_args(['Toy', 'complete', '--max_dist'])
        self.assertRegexpMatches(mock_stderr.getvalue(),
                                 r"expected one argument")

    def test_is_max_dist_defined_true(self):
        self.assertTrue(gb.is_max_dist_defined(0.2))

    def test_is_max_dist_defined_too_small(self):
        self.assertFalse(gb.is_max_dist_defined(0.0))
        self.assertFalse(gb.is_max_dist_defined(-0.1))

    def test_is_max_dist_defined_not_defined(self):
        self.assertFalse(gb.is_max_dist_defined(None))


if __name__ == '__main__':
    main()
