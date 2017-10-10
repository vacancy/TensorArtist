# -*- coding:utf8 -*-
# File   : test_nn_opr_imgproc.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/10/17
# 
# This file is part of TensorArtist.

from tartist.nn import Env, opr as O

import numpy as np
import unittest
import functools


def wraps_new_env(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        with Env().as_default():
            return func(*args, **kwargs)
    return new_func


class TestNNOprImgproc(unittest.TestCase):
    @functools.wraps(np.allclose)
    def assertTensorClose(self, *args, **kwargs):
        return self.assertTrue(np.allclose(*args, **kwargs))

    @wraps_new_env
    def testCropCenter(self):
        a = O.placeholder('a', shape=(16, 17, 17, 3))
        b = O.crop_center(a, [15, 15]) 
        self.assertTupleEqual(b.static_shape, (16, 15, 15, 3))
        
        avar = np.random.normal(size=(16, 17, 17, 3))
        bvar = avar[:, 1:-1, 1:-1, :]
        self.assertTensorClose(b.eval(a=avar), bvar)

    @wraps_new_env
    def testCropLU(self):
        a = O.placeholder('a', shape=(16, 17, 17, 3))
        b = O.crop_lu(a, [15, 15]) 
        self.assertTupleEqual(b.static_shape, (16, 15, 15, 3))
        
        avar = np.random.normal(size=(16, 17, 17, 3))
        bvar = avar[:, :-2, :-2, :]
        self.assertTensorClose(b.eval(a=avar), bvar)

    @wraps_new_env
    def testPaddingCenter(self):
        a = O.placeholder('a', shape=(16, 15, 15, 3))
        b = O.pad_center(a, [17, 17]) 
        self.assertTupleEqual(b.static_shape, (16, 17, 17, 3))
        
        avar = np.random.normal(size=(16, 15, 15, 3))
        bvar = np.pad(avar, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='constant')
        self.assertTensorClose(b.eval(a=avar), bvar)

    @wraps_new_env
    def testPaddingRB(self):
        a = O.placeholder('a', shape=(16, 15, 15, 3))
        b = O.pad_rb(a, [16, 16]) 
        self.assertTupleEqual(b.static_shape, (16, 16, 16, 3))
        
        avar = np.random.normal(size=(16, 15, 15, 3))
        bvar = np.pad(avar, [[0, 0], [0, 1], [0, 1], [0, 0]], mode='constant')
        self.assertTensorClose(b.eval(a=avar), bvar)

    @wraps_new_env
    def testPaddingRBMultiple(self):
        a = O.placeholder('a', shape=(16, 15, 15, 3))
        b = O.pad_rb_multiple_of(a, 8) 
        self.assertTupleEqual(b.static_shape, (16, 16, 16, 3))
        
        avar = np.random.normal(size=(16, 15, 15, 3))
        bvar = np.pad(avar, [[0, 0], [0, 1], [0, 1], [0, 0]], mode='constant')
        self.assertTensorClose(b.eval(a=avar), bvar)
       

if __name__ == '__main__':
    unittest.main()
