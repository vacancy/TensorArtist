# -*- coding:utf8 -*-
# File   : test_core_defaults.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 1/28/17
# 
# This file is part of TensorArtist


import unittest

from tartist.core.utils.defaults import defaults_manager


class TestCoreUtilsDefaults(unittest.TestCase):
    def testDefaults(self):
        class A(object):
            @defaults_manager.wrap_custom_as_default
            def as_default(self):
                yield

        get_default_a = defaults_manager.gen_get_default(A)
        self.assertIsNone(get_default_a())
        a = A()
        with a.as_default():
            self.assertEqual(get_default_a(), a)
            b = A()
            with b.as_default():
                self.assertEqual(get_default_a(), b)
            self.assertEqual(get_default_a(), a)
        self.assertIsNone(get_default_a())


if __name__ == '__main__':
    unittest.main()
