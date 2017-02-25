# -*- coding:utf8 -*-
# File   : cache.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 1/25/17
# 
# This file is part of TensorArtist

import functools

__all__ = ['cached_property', 'cached_result']


class cached_property:
    def __init__(self, fget):
        self.fget = fget
        self.__module__ = fget.__module__
        self.__name__ = fget.__name__
        self.__doc__ = fget.__doc__
        self.__cache_key = '__result_cache_{}_{}'.format(
            fget.__name__, id(fget))

    def __get__(self, instance, owner):
        if instance is None:
            return self.fget
        v = getattr(instance, self.__cache_key, None)
        if v is not None:
            return v
        v = self.fget(instance)
        assert v is not None
        setattr(instance, self.__cache_key, v)
        return v


def cached_result(func):
    def impl():
        nonlocal impl
        ret = func()
        impl = lambda: ret
        return ret

    @functools.wraps(func)
    def f():
        return impl()

    return f

