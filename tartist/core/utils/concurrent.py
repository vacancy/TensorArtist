# -*- coding:utf8 -*-
# File   : concurrent.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/23/17
# 
# This file is part of TensorArtist

import threading
import multiprocessing

__all__ = [
    'MPLibExtension', 'instantiate_mplib_ext',
    'MTBooleanEvent', 'MPBooleanEvent'
]


class MPLibExtension(object):
    __mplib__ = threading


def instantiate_mplib_ext(base_class):
    class MultiThreadingImpl(base_class):
        __name__ = 'MT' + base_class.__name__
        __mplib__ = threading

    class MultiProcessingImpl(base_class):
        __name__ = 'MP' + base_class.__name__
        __mplib__ = multiprocessing

    return MultiThreadingImpl, MultiProcessingImpl


class BooleanEvent(MPLibExtension):
    def __init__(self):
        self._t = type(self).__mplib__.Event()
        self._f = type(self).__mplib__.Event()
        self._t.clear()
        self._f.set()
        self._lock = type(self).__mplib__.Lock()

    def is_true(self):
        return self._t.is_set()

    def is_false(self):
        return self._f.is_set()

    def set(self):
        with self._lock:
            self._t.set()
            self._f.clear()

    def clear(self):
        with self._lock:
            self._t.clear()
            self._f.set()

    def wait(self, predicate=True, timeout=None):
        target = self._t if predicate else self._f
        return target.wait(timeout)

    def wait_true(self, timeout=None):
        return self.wait(True, timeout=timeout)

    def wait_false(self, timeout=None):
        return self.wait(False, timeout=timeout)

    def set_true(self):
        self.set()

    def set_false(self):
        self.clear()

    def value(self):
        return self.is_true()

MTBooleanEvent, MPBooleanEvent = instantiate_mplib_ext(BooleanEvent)

