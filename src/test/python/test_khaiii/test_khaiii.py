#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
khaiii tests
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2018-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
import unittest

import khaiii    # pylint: disable=import-error
from khaiii import KhaiiiExcept    # pylint: disable=import-error


#########
# tests #
#########
class TestKhaiii(unittest.TestCase):
    """
    khaiii tests
    """
    def setUp(self):
        self._api = khaiii.KhaiiiApi()
        self._api.set_log_level('all', 'warn')

    def tearDown(self):
        self._api.close()

    def test_version(self):
        """
        test version() api
        """
        self.assertRegex(self._api.version(), r'^\d+\.\d+(\.\d+)?$')

    def test_open(self):
        """
        test open() api
        """
        try:
            self._api.open()
        except KhaiiiExcept as khaiii_exc:
            self.fail(khaiii_exc)
        with self.assertRaises(KhaiiiExcept):
            self._api.open('/not/existing/dir')
        with self.assertRaises(KhaiiiExcept):
            self._api.open('', 'invalid option')

    def test_analyze(self):
        """
        test analyze() api
        """
        try:
            words = self._api.analyze('나는 학교에 갔다.')
            self.assertEqual(len(words), 3)
            self.assertEqual(len(words[0].morphs), 2)
            self.assertEqual(words[0].morphs[0].lex, '나')
            self.assertEqual(words[0].morphs[0].tag, 'NP')
            self.assertEqual(words[0].morphs[1].lex, '는')
            self.assertEqual(words[0].morphs[1].tag, 'JX')
            self.assertEqual(len(words[1].morphs), 2)
            self.assertEqual(words[1].morphs[0].lex, '학교')
            self.assertEqual(words[1].morphs[0].tag, 'NNG')
            self.assertEqual(words[1].morphs[1].lex, '에')
            self.assertEqual(words[1].morphs[1].tag, 'JKB')
            self.assertEqual(len(words[2].morphs), 4)
            self.assertEqual(words[2].morphs[0].lex, '가')
            self.assertEqual(words[2].morphs[0].tag, 'VV')
            self.assertEqual(words[2].morphs[1].lex, '았')
            self.assertEqual(words[2].morphs[1].tag, 'EP')
            self.assertEqual(words[2].morphs[2].lex, '다')
            self.assertEqual(words[2].morphs[2].tag, 'EF')
            self.assertEqual(words[2].morphs[3].lex, '.')
            self.assertEqual(words[2].morphs[3].tag, 'SF')
        except KhaiiiExcept as khaiii_exc:
            self.fail(khaiii_exc)

    def test_analyze_bfr_errpatch(self):
        """
        test analyze_bfr_errpatch() api
        """
        try:
            results = self._api.analyze_bfr_errpatch('테스트')
            self.assertEqual(len(results), len('테스트') + 2)
        except KhaiiiExcept as khaiii_exc:
            self.fail(khaiii_exc)

    def test_set_log_level(self):
        """
        test set_log_level() api
        """
        try:
            self._api.set_log_level('all', 'info')
        except KhaiiiExcept as khaiii_exc:
            self.fail(khaiii_exc)
        with self.assertRaises(KhaiiiExcept):
            self._api.set_log_level('all', 'not_existing_level')


########
# main #
########
if __name__ == '__main__':
    unittest.main()
