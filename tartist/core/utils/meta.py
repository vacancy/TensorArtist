# -*- coding:utf8 -*-
# File   : meta.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/29/16
# 
# This file is part of TensorArtist


import collections

import numpy

__all__ = ['iter_kv',
           'dict_deep_update', 'dict_deep_keys',
           'astuple', 'asshape',
           'assert_instance', 'assert_none', 'assert_notnone',
           'UniqueValueGetter']


def iter_kv(v):
    if isinstance(v, dict):
        return v.items()
    assert_instance(v, collections.Iterable)
    return enumerate(v)


def dict_deep_update(a, b):
    """
    A deep update function for python's internal dict

    It is used to replace the traditional: a.update(b)

    :param a: the result
    :param b: the dict to be added to a
    :return: None
    """
    for key in b:
        if key in a and type(b[key]) is dict:
            dict_deep_update(a[key], b[key])
        else:
            a[key] = b[key]


def dict_deep_keys(d, sort=True, split='.'):
    assert type(d) is dict

    def _dfs(current, result, prefix=None):
        for key in current:
            current_key = key if prefix is None else '{}{}{}'.format(prefix, split, key)
            result.append(current_key)
            if type(current[key]) is dict:
                _dfs(current[key], res, current_key)

    res = list()
    _dfs(d, res)
    if sort:
        res.sort()
    return res


def astuple(arr_like):
    if type(arr_like) is tuple:
        return arr_like
    elif type(arr_like) in (bool, int, float):
        return tuple((arr_like,))
    else:
        return tuple(arr_like)


def asshape(arr_like):
    if type(arr_like) is tuple:
        return arr_like
    elif type(arr_like) is int:
        if arr_like == 0:
            return tuple()
        else:
            return tuple((arr_like,))
    elif arr_like is None:
        return None,
    else:
        return tuple(arr_like)


def assert_instance(ins, clz, msg=None):
    msg = msg or '{} is not of class {}'.format(ins, clz)
    assert isinstance(ins, clz), msg


def assert_none(ins, msg=None):
    msg = msg or '{} is not None'.format(ins)
    assert ins is None, msg


def assert_notnone(ins, msg=None, name='instance'):
    msg = msg or '{} is None'.format(name)
    assert ins is not None, msg


class UniqueValueGetter(object):
    def __init__(self, msg='unique value check failed', default=None):
        self._msg = msg
        self._val = None
        self._default = default

    def set(self, v):
        assert self._val is None or self._val == v, self._msg + ': expect={} got={}'.format(self._val, v)
        self._val = v

    def get(self):
        return self._val or self._default

