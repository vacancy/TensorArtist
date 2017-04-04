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
    def __init__(self, kvstore):
        self._kvstore = kvstore
        self._keys = self._kvstore.keys()

    def _gen(self):
        for k in self._keys:
            yield self._kvstore.get(k)


class KVStoreRandomSampleDataFlow(RandomizedDataFlowBase):
    def __init__(self, kvstore, seed=None):
        super().__init__(seed=seed)
        self._kvstore = kvstore
        self._keys = kvstore.keys()

    def _gen(self):
        while True:
            k = self._rng.choice(self._keys)
            yield self._kvstore.get(k)

