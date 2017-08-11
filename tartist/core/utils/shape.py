# -*- coding:utf8 -*-
# File   : shape.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/14/17
# 
# This file is part of TensorArtist.

import collections

__all__ = ['get_2dshape', 'get_4dshape']


def get_2dshape(x, default=None, type=int):
    if x is None:
        return default
    if isinstance(x, collections.Iterable):
        x = tuple(x)
        if len(x) == 1:
            return x[0], x[0]
        else:
            assert len(x) == 2, '2dshape must be of length 1 or 2'
            return x
    else:
        x = type(x)
        return x, x


def get_4dshape(x, default=None, type=int):
    if x is None:
        return default
    if isinstance(x, collections.Iterable):
        x = tuple(x)
        if len(x) == 1:
            return 1, x[0], x[0], 1
        elif len(x) == 2:
            return 1, x[0], x[1], 1
        else:
            assert len(x) == 4, '4dshape must be of length 1, 2, or 4'
            return x
    else:
        x = type(x)
        return 1, x, x, 1
