# -*- coding:utf8 -*-
# File   : rng.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/23/17
#
# This file is part of TensorArtist.
# This file is part of NeuArtist2.

import os
import threading
import contextlib
import random as _random

import numpy as np
import numpy.random as npr

__all__ = [
    'get_rng', 'with_rng', 'reset_rng', 
    'gen_seed', 'gen_rng', 
    'shuffle_multiarray', 'list_choice', 'list_shuffle'
]

_rng = None
_rng_local = threading.local()


def get_rng():
    if getattr(_rng_local, 'rng', None) is None:
        return _rng
    return _rng_local.rng


def reset_rng(seed=None):
    """Reset the program-level random generator, non-threading-safe function."""
    global _rng
    if _rng is None:
        _rng = npr.RandomState(seed)
    else:
        rng2 = npr.RandomState(seed)
        _rng.set_state(rng2.get_state())


@contextlib.contextmanager
def with_rng(rng):
    old_rng = getattr(_rng_local, 'rng', None)
    _rng_local.rng = rng
    yield rng
    _rng_local.rng = old_rng


def gen_seed():
    return _rng.randint(4294967296)


def gen_rng(seed=None):
    if seed is None:
        seed = gen_seed()
    return npr.RandomState(seed)


def shuffle_multiarray(*arrs, rng=None):
    rng = rng or _rng
    length = len(arrs[0])
    for a in arrs:
        assert len(a) == length, 'non-compatible length when shuffling multiple arrays'

    inds = np.arange(length)    
    rng.shuffle(inds)
    return tuple(map(lambda x: x[inds], arrs))


def list_choice(l, rng=None):
    """Efficiently draw an element from an list, if the rng is given, use it instead of the system one."""
    rng = rng or _rng
    assert type(l) in (list, tuple)
    return l[rng.choice(len(l))]


def list_shuffle(l, rng=None):
    rng = rng or _rng
    if isinstance(l, np.ndarray):
        rng.shuffle(l)
        return

    assert type(l) is list
    _random.shuffle(l, random=rng.random_sample)


def __initialize_rng():
    seed = os.getenv('TART_RANDOM_SEED')
    reset_rng(seed)


__initialize_rng()
