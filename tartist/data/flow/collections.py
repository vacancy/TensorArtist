# -*- coding:utf8 -*-
# File   : collections.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/23/17
# 
# This file is part of TensorArtist.

from .base import SimpleDataFlowBase, ProxyDataFlowBase
from ...core.utils.meta import UniqueValueGetter
import numpy as np

__all__ = [
    'DictDataFlowProxy', 'EmptyDictDataFlow',
    'QueueDataFlow', 'PoolDataFlow',
    'ListOfArrayDataFlow', 'DictOfArrayDataFlow',
    'DictToBatchDataFlow'
]


class DictDataFlowProxy(ProxyDataFlowBase):
    def __init__(self, keys, iterable):
        super().__init__(iterable)
        self._keys = keys
        self._iterable = iterable
   
    def _gen(self):
        for v in self._iterable:
            assert len(self._keys) == len(v), 'DictDataFlowAdapter: length mismatched'
            yield dict(zip(self._keys, v))


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


class PoolDataFlow(SimpleDataFlowBase):
    def __init__(self, pool):
        self._pool = pool
        self._length = len(self._pool)

    def _gen(self):
        for i in range(self._length):
            yield self._pool[i]

    def _len(self):
        return self._length


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


class DictToBatchDataFlow(ProxyDataFlowBase):
    def __init__(self, iterable, excludes=None):
        super().__init__(iterable)
        self._excludes = set(excludes) if excludes is not None else set()

    def _gen(self):
        for item in self.unwrapped:
            for k, v in item.items():
                if k not in self._excludes:
                    item[k] = np.array(v)[np.newaxis]
