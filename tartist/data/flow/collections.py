# -*- coding:utf8 -*-
# File   : collections.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/23/17
# 
# This file is part of TensorArtist

from .base import SimpleDataFlowBase
from .rng import RandomizedDataFlowBase
from ...core.utils.meta import UniqueValueGetter
import collections

__all__ = ['DictDataFlowProxy', 'EmptyDictDataFlow', 
        'QueueDataFlow', 'RandomRepeatDataFlow',
        'ListOfArrayDataFlow', 'DictOfArrayDataFlow']


class DictDataFlowProxy(SimpleDataFlowBase):
    def __init__(self, keys, iterable):
        self._keys = keys
        self._iterable = iterable
   
    def _gen(self):
        for v in self._iterable:
            assert len(self._keys) == len(v), 'DictDataFlowAdapter: length mismatched'
            yield dict(zip(self._keys, v))

    def _len(self):
        return len(self._iterable)


class EmptyDictDataFlow(SimpleDataFlowBase):
    def _gen(self):
        while True:
            yield {}


class QueueDataFlow(SimpleDataFlowBase):
    def __init__(self, queue):
        self._queue = queue

    def _gen(self):
        while True:
            yield self._queue.get()


class RandomRepeatDataFlow(RandomizedDataFlowBase):
    def __init__(self, source, nr_repeat, cache_size, seed=None):
        super().__init__(seed=seed)
        self._source = source
        self._nr_repeat = nr_repeat
        self._cache_size = cache_size

    def _gen(self):
        it = iter(self._source)
        while True:
            data = []
            for i in range(self._cache_size):
                d = next(it)
                data.append(d)
                yield d
            for i in range((self._nr_repeat - 1) * self._cache_size):
                idx = self._rng.randint(len(data))
                yield data[idx]


class ListOfArrayDataFlow(SimpleDataFlowBase):
    def __init__(self, loa):
        self._loa = loa

        uvg = UniqueValueGetter('ListOfArrayDataFlow length consistency check failed')
        for i in self._loa:
            uvg.set(len(i))
        self._length = uvg.get()

    def _gen(self):
        for i in range(self._length):
            yield [l[i] for l in self._loa]

    def _len(self):
        return self._length


def DictOfArrayDataFlow(doa):
    keys = doa.keys()
    values = [doa[k] for k in keys]
    return DictDataFlowProxy(keys, ListOfArrayDataFlow(values))

