# -*- coding:utf8 -*-
# File   : kv.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/30/17
# 
# This file is part of TensorArtist

from .base import SimpleDataFlowBase
from .rng import RandomizedDataFlowBase 

__all__ = ['KVStoreDataFlow', 'KVStoreRandomSampleDataFlow']


class KVStoreDataFlow(SimpleDataFlowBase):
    def __init__(self, kv_getter):
        self._kv_getter = kv_getter
        self._kvstore = None
        self._keys = None

    def _initialize(self):
        super()._initialize()
        self._kvstore = self._kv_getter()
        self._keys = list(self._kvstore.keys())

    def _gen(self):
        for k in self._keys:
            yield self._kvstore.get(k)


class KVStoreRandomSampleDataFlow(RandomizedDataFlowBase):
    def __init__(self, kv_getter, seed=None):
        super().__init__(seed=seed)
        self._kv_getter = kv_getter
        self._kvstore = None
        self._keys = None
        self._nr_keys = None

    def _initialize(self):
        super()._initialize()
        self._kvstore = self._kv_getter()
        self._keys = list(self._kvstore.keys())
        self._nr_keys = len(self._keys)

    def _gen(self):
        while True:
            k = self._keys[self._rng.choice(self._nr_keys)]
            yield self._kvstore.get(k)

