# -*- coding:utf8 -*-
# File   : test_random_mt.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/11/17
# 
# This file is part of TensorArtist.

from tartist import random as tar
from tartist.core.utils.meta import map_exec

import threading
import queue
import time
import unittest
import contextlib
import numpy as np

from threading import Thread


class TestRandomMT(unittest.TestCase):
    def testMTRandomness(self):
        q = queue.Queue() 

        def proc():
            rng = tar.gen_rng()
            with tar.with_rng(rng):
                time.sleep(0.5)
                state = tar.get_rng().get_state()
                time.sleep(0.5)
                q.put(state)

        threads = [Thread(target=proc) for i in range(2)]
        map_exec(Thread.start, threads)
        map_exec(Thread.join, threads)

        v1, v2 = q.get(), q.get()
        self.assertFalse(np.allclose(v1[1], v2[1]))
    
    def testFakeMTRandomness(self):
        mutex = threading.Lock()

        @contextlib.contextmanager
        def fake_with_rng(rrr):
            from tartist.random import rng
            with mutex:
                backup = rng._rng
                rng._rng = rrr
            yield rrr
            with mutex:
                rng._rng = backup

        q = queue.Queue() 

        def proc():
            rng = tar.gen_rng()
            with fake_with_rng(rng):
                time.sleep(0.5)
                state = tar.get_rng().get_state()
                time.sleep(0.5)
                q.put(state)

        threads = [Thread(target=proc) for i in range(2)]
        map_exec(Thread.start, threads)
        map_exec(Thread.join, threads)

        v1, v2 = q.get(), q.get()
        self.assertFalse(not np.allclose(v1[1], v2[1]))


if __name__ == '__main__':
    unittest.main()
