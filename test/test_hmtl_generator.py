from unittest import main, TestCase

import pandas as pd

import aggregation.html_generator as hg


class TestHTMLGenerator(TestCase):
    def test_render_dictionary_in_html(self):
        """ Should generate the correct HTML file (snapshot test). """
        dictionary = pd.DataFrame({'token': ['a', 'b'],
                                   'token_id': [[0, 2, 4], [1, 3]],
                                   'sentence_id': [[0, 1, 2], [0, 2]],
                                   'sense': [['a0', 'a1', 'a0'], ['b0', 'b0']]})
        sentences = [['a', 'b'], ['a'], ['b', 'a']]

        html_expected = \
            '<!DOCTYPE html>\n<html>\n<head>\n' \
            '<title>Exp007</title>\n</head>\n<body>\n' \
            '<h1>Exp007</h1>\n<hr>\n<h2>Tokens</h2>\n' \
            '<a href="#a">a</a>,\n<a href="#b">b</a>\n' \
            '<hr>\n<h2>Dictionary</h2>\n' \
            '<h3 id="a">a</h3>\n<ul>\n<li id="a0"><b>a0: </b>\n' \
            '<a href="#S0">S0</a>, <a href="#S2">S2</a>\n</li>' \
            '\n<li id="a1"><b>a1: </b>\n<a href="#S1">S1</a>\n</li>\n' \
            '</ul>\n<h3 id="b">b</h3>\n<ul>\n<li id="b0"><b>b0: </b>\n' \
            '<a href="#S0">S0</a>, <a href="#S2">S2</a>\n</li></ul>\n' \
            '<hr>\n<h2>Sentences</h2>\n<ul>\n<li><b id="S0">S0: </b>' \
            '<a href="#a">a</a> <a href="#b">b</a></li>\n' \
            '<li><b id="S1">S1: </b><a href="#a">a</a></li>\n<li>' \
            '<b id="S2">S2: </b><a href="#b">b</a> <a href="#a">a</a>' \
            '</li>\n</ul>\n</body>\n</html>\n'
        html_result = hg.render_dictionary_in_html(
            dictionary, sentences, 'Exp007')

        self.assertEqual(html_expected, html_result)


if __name__ == '__main__':
    main()
