# -*- coding:utf8 -*-
# File   : defaults.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 1/28/17
# 
# This file is part of TensorArtist


import contextlib
import functools


class DefaultsManager(object):
    def __init__(self):
        self._defaults = {}

    @staticmethod
    def __make_unique_identifier(func):
        module = func.__module__
        qualname = func.__qualname__
        assert '.' in qualname, 'invalid qualname for function {}'.format(func)
        qualname = qualname[:qualname.rfind('.')]
        return '{}.{}'.format(module, qualname)

    def wrap_custom_as_default(self, custom_method):
        identifier = self.__make_unique_identifier(custom_method)

        @contextlib.contextmanager
        @functools.wraps(custom_method)
        def wrapped_func(slf, *args, **kwargs):
            backup = self._defaults.get(identifier, None)
            self._defaults[identifier] = slf
            yield custom_method(slf, *args, **kwargs)
            self._defaults[identifier] = backup

        return wrapped_func

    def gen_get_default(self, cls):
        identifier = self.__make_unique_identifier(cls.as_default)

        def get_default(default=None):
            return self._defaults.get(identifier, default)
        return get_default

defaults_manager = DefaultsManager()
