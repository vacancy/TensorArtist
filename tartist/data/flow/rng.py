# -*- coding:utf8 -*-
# File   : rng.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/23/17
# 
# This file is part of TensorArtist.

from .base import SimpleDataFlowBase
from ...random import gen_rng, list_shuffle
from ...core.utils.meta import UniqueValueGetter


__all__ = [
    'RandomizedDataFlowBase',
    'PoolRandomSampleDataFlow',
    'LOARandomSampleDataFlow',
    'DOARandomSampleDataFlow',
    'RandomRepeatDataFlow'
]


class RandomizedDataFlowBase(SimpleDataFlowBase):
    _rng = None

    def __init__(self, seed=None):
        self._seed = seed

    def _initialize(self):
        self._rng = gen_rng(seed=self._seed)


class PoolRandomSampleDataFlow(RandomizedDataFlowBase):
    _pool = None

    def __init__(self, pool, seed=None):
        super().__init__(seed=seed)
        self._pool = pool
        self._length = len(self._pool)

    def _gen(self):
        while True:
            list_shuffle(self._pool, self._rng)
            for i in range(self._length):
                yield self._pool[i]


class LOARandomSampleDataFlow(RandomizedDataFlowBase):
    _loa = None
    _length = None

    def __init__(self, loa, seed=None):
        super().__init__(seed=seed)
        self._set_loa(loa)

    def _set_loa(self, loa):
        self._loa = loa
        uvg = UniqueValueGetter('LOARandomSampleDataFlow length consistency check failed')
        for i in self._loa:
            uvg.set(len(i))
        self._length = uvg.get()

    def _gen(self):
        while True:
            state = self._rng.get_state()
            for item in self._loa:
                self._rng.set_state(state)
                list_shuffle(item, self._rng)
            for i in range(self._length):
                yield [l[i] for l in self._loa]


def DOARandomSampleDataFlow(doa, seed=None):
    from .collections import DictDataFlowProxy

    keys = doa.keys()
    values = [doa[k] for k in keys]
    return DictDataFlowProxy(keys, LOARandomSampleDataFlow(values, seed=seed))


class RandomRepeatDataFlow(RandomizedDataFlowBase):
    def __init__(self, source, nr_repeat, cache_size, block=False, seed=None):
        super().__init__(seed=seed)
        self._source = source
        self._nr_repeat = nr_repeat
        self._cache_size = cache_size
        self._block = block

    def _gen(self):
        it = iter(self._source)
        while True:
            data = []
            for i in range(self._cache_size):
                d = next(it)
                data.append(d)
                if not self._block:
                    yield d

            nr_repeat = self._nr_repeat if self._block else self._nr_repeat - 1
            for i in range(nr_repeat * self._cache_size):
                idx = self._rng.randint(len(data))
                yield data[idx]
