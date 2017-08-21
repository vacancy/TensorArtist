# -*- coding:utf8 -*-
# File   : sampler.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 17/08/2017
# 
# This file is part of TensorArtist.

from .rng import gen_rng

__all__ = ['SimpleBatchSampler']


class SimpleBatchSampler(object):
    def __init__(self, batch_size, nr_repeat, rng=None):
        self._batch_size = batch_size
        self._nr_repeat = nr_repeat
        self._rng = rng or gen_rng()

    def _gen(self, data, keys):
        n = len(data[keys[0]])
        print(n)

        for i in range(self._nr_repeat):
            idx = self._rng.permutation(n)
            for j in range(n // self._batch_size):
                this = {
                    k: data[k][idx[j * self._batch_size:j * self._batch_size + self._batch_size]]
                    for k in keys
                }
                yield this

    def __call__(self, data, keys):
        return self._gen(data, keys)
