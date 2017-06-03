# -*- coding:utf8 -*-
# File   : nd.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 1/30/17
# 
# This file is part of TensorArtist.

import collections
import numpy

__all__ = [
    'isndarray', 
    'nd_concat', 'nd_len', 'nd_batch_size', 
    'nd_split_n', 'size_split_n'
]


def isndarray(arr):
    return isinstance(arr, numpy.ndarray)


def nd_concat(lst):
    if len(lst) == 0:
        return None
    elif len(lst) == 1:
        return lst[0]
    else:
        return numpy.concatenate(lst)


def nd_len(arr):
    if type(arr) in (int, float):
        return 1
    if isndarray(arr):
        return arr.shape[0]
    return len(arr)


def nd_batch_size(thing):
    if type(thing) in (tuple, list):
        return nd_len(thing[0])
    elif type(thing) in (dict, collections.OrderedDict):
        return nd_len(list(thing.values())[0])
    else:
        raise NotImplementedError()


def nd_split_n(ndarray, n):
    sub_sizes = size_split_n(len(ndarray), n)
    res = []
    cur = 0
    for i in range(n):
        res.append(ndarray[cur:cur+sub_sizes[i]])
        cur += sub_sizes[i]
    return res


def size_split_n(full_size, n):
    if full_size is None:
        return None
    result = [full_size // n] * n
    rest = full_size % n
    if rest != 0:
        result[-1] += rest
    return result
