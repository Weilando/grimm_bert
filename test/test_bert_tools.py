import os.path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

import analysis.bert_tools as abt


class TestBertTools(TestCase):
    def test_should_download_model(self):
        with TemporaryDirectory() as tmp_dir_name:
            self.assertTrue(abt.should_download_model(tmp_dir_name, 'model'))

    def test_should_not_download_model(self):
        with TemporaryDirectory() as tmp_dir_name:
            os.mkdir(os.path.join(tmp_dir_name, 'model'))
            self.assertFalse(abt.should_download_model(tmp_dir_name, 'model'))


if __name__ == '__main__':
    main()
