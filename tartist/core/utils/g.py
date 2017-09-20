# -*- coding:utf8 -*-
# File   : g.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 20/09/2017
# 
# This file is part of TensorArtist.

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

g = G()
