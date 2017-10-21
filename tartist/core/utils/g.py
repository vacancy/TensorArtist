# -*- coding:utf8 -*-
# File   : g.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 20/09/2017
# 
# This file is part of TensorArtist.

import sys

__all__ = ['G', 'g']


class G(dict):
    def __getattr__(self, k):
        if k not in self:
            raise AttributeError
        return self[k]

    def __setattr__(self, k, v):
        self[k] = v

    def __delattr__(self, k):
        del self[k]

    def print(self, sep=': ', end='\n', file=None):
        keys = sorted(self.keys())
        lens = list(map(len, keys))
        max_len = max(lens)
        for k in keys:
            print(k + ' ' * (max_len - len(k)), self[k], sep=sep, end=end, file=file, flush=True)

g = G()
