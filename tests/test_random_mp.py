# -*- coding:utf8 -*-
# File   : test_random_mp.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/23/17
# 
# This file is part of TensorArtist.

from tartist import random as tar

from multiprocessing import Process
import multiprocessing
import unittest


class TestRandomMP(unittest.TestCase):
    def testMPRandomness(self):
        q = multiprocessing.Queue()

        def proc():
            tar.reset_rng()
            v2 = tar.normal()
            q.put(v2)

        p = Process(target=proc)
        p.start()
        p.join()
        v2 = q.get()

        v1 = tar.normal()
        print(v1, v2)
        self.assertNotEqual(v1, v2)


if __name__ == '__main__':
    unittest.main()
