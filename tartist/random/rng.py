# -*- coding:utf8 -*-
# File   : rng.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/23/17
#
# This file is part of TensorArtist
# This file is part of NeuArtist2

import os

import numpy as np
import numpy.random as npr

__all__ = ['rng', 'reset_rng', 'gen_seed', 'gen_rng', 'shuffle_multiarray']

rng = None


def __initialize_rng():
    seed = os.getenv('TART_RANDOM_SEED')
    reset_rng(seed)


def reset_rng(seed=None):
    global rng
    if rng is None:
        rng = npr.RandomState(seed)
    else:
        rng2 = npr.RandomState(seed)
        rng.set_state(rng2.get_state())


def gen_seed():
    global rng
    return rng.randint(4294967296)


def gen_rng(seed=None):
    if seed is None:
        seed = gen_seed()
    return npr.RandomState(seed)


def shuffle_multiarray(*arrs):
    length = len(arrs[0])
    for a in arrs:
        assert len(a) == length, 'non-compatible length when shuffling multiple arrays'

    inds = np.arange(length)    
    rng.shuffle(inds)
    return tuple(map(lambda x: x[inds], arrs))

__initialize_rng()
