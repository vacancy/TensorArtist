# -*- coding:utf8 -*-
# File   : rng.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/23/17
# 
# This file is part of TensorArtist

from .base import SimpleDataFlowBase
from .collections import DictDataFlowProxy
from ...random import gen_rng
from ...core.utils.meta import UniqueValueGetter

import numpy as np

__all__ = ['RandomizedDataFlowBase', 
        'LOARandomSampleDataFlow', 
        'DOARandomSampleDataFlow']


class RandomizedDataFlowBase(SimpleDataFlowBase):
    _rng = None

    def __init__(self, seed=None):
        self._seed = seed

    def _initialize(self):
        self._rng = gen_rng(seed=self._seed)


class LOARandomSampleDataFlow(RandomizedDataFlowBase):
    def __init__(self, loa, seed=None):
        super().__init__(seed=seed)
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
                self._rng.shuffle(item)
            for i in range(self._length):
                yield [l[i] for l in self._loa]


def DOARandomSampleDataFlow(doa, seed=None):
    keys = doa.keys()
    values = [doa[k] for k in keys]
    return DictDataFlowProxy(keys, LOARandomSampleDataFlow(values, seed=seed))

