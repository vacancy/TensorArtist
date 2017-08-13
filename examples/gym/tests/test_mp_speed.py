# -*- coding:utf8 -*-
# File   : test_mp_speed.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/08/2017
# 
# This file is part of TensorArtist.

from tartist import image
from tartist.app import rl

import time
import multiprocessing.pool as mppool


def make_player(dump_dir=None):
    def resize_state(s):
        return image.resize(s, (84, 84), interpolation='NEAREST')

    p = rl.GymRLEnviron('Enduro-v0', dump_dir=dump_dir)
    p = rl.MapStateProxyRLEnviron(p, resize_state)
    p = rl.HistoryFrameProxyRLEnviron(p, 4)
    p = rl.LimitLengthProxyRLEnviron(p, 4000)
    return p


def actor(s):
    return 1


def worker(i):
    p = make_player()
    l = 0
    for i in range(1):
        p.play_one_episode(func=actor)
        l += p.stats['length'][-1]
    return l


def test_mp():
    pool = mppool.Pool(4)
    start_time = time.time()
    lengths = pool.map(worker, range(4))
    finish_time = time.time()
    print('Multiprocessing: total_length={}, time={:.2f}s.'.format(sum(lengths), finish_time - start_time))


def test_mt():
    pool = mppool.ThreadPool(4)
    start_time = time.time()
    lengths = pool.map(worker, range(4))
    finish_time = time.time()
    print('Multithreading: total_length={}, time={:.2f}s.'.format(sum(lengths), finish_time - start_time))


if __name__ == '__main__':
    test_mp()
    test_mt()
