# -*- coding:utf8 -*-
# File   : test_core_defaults.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 1/28/17
# 
# This file is part of TensorArtist.


import unittest

from tartist.core import with_env, load_env, get_env, set_env


class TestCoreEnviron(unittest.TestCase):
    def testWithEnv(self):
        env_a = {'value': 1}
        env_b = {'value': 2}

        load_env(env_a)
        set_env('value', 1.1)
        self.assertEqual(get_env('value'), 1.1)

        with with_env(env_b):
            self.assertEqual(get_env('value'), 2)
            set_env('value', 2.1)
            self.assertEqual(get_env('value'), 2.1)
        
        self.assertEqual(get_env('value'), 1.1)

        load_env(env_a)
        self.assertEqual(get_env('value'), 1)
        load_env(env_b)
        self.assertEqual(get_env('value'), 2)


if __name__ == '__main__':
    unittest.main()
