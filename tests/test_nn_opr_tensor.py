# -*- coding:utf8 -*-
# File   : test_nn_opr_tensor.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/23/17
# 
# This file is part of TensorArtist.

from tartist.nn import Env, opr as O

import numpy as np
import unittest
import functools


class TestNNOprTensor(unittest.TestCase):
    @functools.wraps(np.allclose)
    def assertTensorClose(self, *args, **kwargs):
        return self.assertTrue(np.allclose(*args, **kwargs))

    def testAdvancedIndexing(self):
        a = O.placeholder('a', shape=(5, 5))
        a_val = np.arange(25).reshape((5, 5)).astype('float32')
        feed_dict = {a.name: a_val}

        self.assertTensorClose(a[0:3].eval(feed_dict=feed_dict), a_val[0:3])
        self.assertTensorClose(a[0:3, 0:3].eval(feed_dict=feed_dict), a_val[0:3, 0:3])
        with self.assertRaises(NotImplementedError):
            self.assertTensorClose(a.set_sub[0:3](1).eval(feed_dict=feed_dict), np.array([1, 1, 1, 3, 4]))
        if True:
            self.assertTensorClose(a.ai[[0, 3]].eval(feed_dict=feed_dict), a_val[[0, 3]])
            self.assertTensorClose(a.ai[[0, 3], [0, 3]].eval(feed_dict=feed_dict), a_val[[0, 3], [0, 3]])
        with self.assertRaises(NotImplementedError):
            self.assertTensorClose(a.set_ai[[0, 3]](1).eval(feed_dict=feed_dict), np.array([1, 1, 1, 3, 4]))


if __name__ == '__main__':
    unittest.main()
