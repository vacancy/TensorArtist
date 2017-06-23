# -*- coding:utf8 -*-
# File   : meta.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/29/16
# 
# This file is part of TensorArtist.


import functools
import collections

import numpy

__all__ = ['iter_kv', 'merge_iterable',
           'dict_deep_update', 'dict_deep_keys',
           'astuple', 'asshape',
           'canonize_args_list',
           'assert_instance', 'assert_none', 'assert_notnone',
           'notnone_property',
           'UniqueValueGetter', 'AttrObject',
           'run_once']


def iter_kv(v):
    if isinstance(v, dict):
        return v.items()
    assert_instance(v, collections.Iterable)
    return enumerate(v)


def merge_iterable(v1, v2):
    assert issubclass(type(v1), type(v2)) or issubclass(type(v2), type(v1))
    if isinstance(v1, (dict, set)):
        v = v1.copy().update(v2)
        return v

    return v1 + v2


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
    elif isinstance(arr_like, collections.Iterable):
        return tuple(arr_like)
    else:
        return tuple((arr_like,))


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


def canonize_args_list(args, *, allow_empty=False, cvt=None):
    if not allow_empty and not args:
        raise TypeError('at least one argument must be provided')

    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        args = args[0]
    if cvt is not None:
        args = tuple(map(cvt, args))
    return args


def assert_instance(ins, clz, msg=None):
    msg = msg or '{} (of type{}) is not of type {}'.format(ins, type(ins), clz)
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


class AttrObject(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __setattr__(self, k, v):
        assert not k.startswith('_')
        if k not in type(self).__dict__:
            # do not use hasattr; since it may result in infinite recursion
            raise AttributeError(
                '{}: could not set non-existing attribute {}'.format(
                    self, k))
        cvt = getattr(type(self), '_convert_{}'.format(k), None)
        if cvt is not None:
            v = cvt(v)
        super().__setattr__(k, v)


class notnone_property:
    def __init__(self, fget):
        self.fget = fget
        self.__module__ = fget.__module__
        self.__name__ = fget.__name__
        self.__doc__ = fget.__doc__
        self.__prop_key  = '{}_{}'.format(
            fget.__name__, id(fget))

    def __get__(self, instance, owner):
        if instance is None:
            return self.fget
        v = self.fget(instance)
        assert v is not None, '{}.{} can not be None, maybe not set yet'.format(
                type(instance).__name__, self.__name__)
        return v


def run_once(func):
    has_run = False

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        nonlocal has_run
        if not has_run:
            has_run = True
            return func(*args, **kwargs)
        else:
            return
    return new_func
