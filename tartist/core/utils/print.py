# -*- coding:utf8 -*-
# File   : print.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 5/4/17
# 
# This file is part of TensorArtist.

import io
import sys
import numpy as np
import collections
import threading

__all__ = ['stprint', 'stformat', 'print_to_string']

__stprint_locks = collections.defaultdict(threading.Lock)


def _indent_print(msg, indent, prefix=None, end='\n', file=sys.stdout):
    print(*['  '] * indent, end='', file=file)
    if prefix is not None:
        print(prefix, end='', file=file)
    print(msg, end=end, file=file)


def stprint(data, key=None, indent=0, file=None, need_lock=True):
    """
    Structure print. Usage:

    ```
    data = dict(a=np.zeros(shape=(10, 10)), b=3)
    stprint(data)
    ```

    and you will get:

    ```
    dict{
        a: ndarray(10, 10), dtype=float64
        b: 3
    }
    ```

    :param data: Data you want to print.
    :param key: Output prefix, internal usage only.
    :param indent: Indent level of the print, internal usage only.
    """
    t = type(data)

    if need_lock:
        __stprint_locks[file].acquire()

    if t is tuple:
        _indent_print('tuple[', indent, prefix=key, file=file)
        for v in data:
            stprint(v, indent=indent + 1, file=file, need_lock=False)
        _indent_print(']', indent, file=file)
    elif t is list:
        _indent_print('list[', indent, prefix=key, file=file)
        for v in data:
            stprint(v, indent=indent + 1, file=file, need_lock=False)
        _indent_print(']', indent, file=file)
    elif t is dict:
        _indent_print('dict{', indent, prefix=key, file=file)
        for k in sorted(data.keys()):
            v = data[k]
            stprint(v, indent=indent + 1, key='{}: '.format(k), file=file, need_lock=False)
        _indent_print('}', indent, file=file)
    elif t is np.ndarray:
        _indent_print('ndarray{}, dtype={}'.format(data.shape, data.dtype), indent, prefix=key, file=file)
    else:
        _indent_print(data, indent, prefix=key, file=file)

    if need_lock:
        __stprint_locks[file].release()


def stformat(data, key=None, indent=0):
    f = io.StringIO()
    stprint(data, key=key, indent=indent, file=f, need_lock=False)
    value = f.getvalue()
    f.close()
    return value


class _PrintToStringContext(object):
    __global_locks = collections.defaultdict(threading.Lock)

    def __init__(self, target='STDOUT', need_lock=True):
        assert target in ('STDOUT', 'STDERR')
        self._target = target
        self._need_lock = need_lock
        self._stream = io.StringIO()
        self._backup = None
        self._value = None

    def _swap(self, rhs):
        if self._target == 'STDOUT':
            sys.stdout, rhs = rhs, sys.stdout
        else:
            sys.stderr, rhs = rhs, sys.stderr

        return rhs

    def __enter__(self):
        if self._need_lock:
            self.__global_locks[self._target].acquire()
        self._backup = self._swap(self._stream)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stream = self._swap(self._backup)
        if self._need_lock:
            self.__global_locks[self._target].release()

    def _ensure_value(self):
        if self._value is None:
            self._value = self._stream.getvalue()
            self._stream.close()

    def get(self):
        self._ensure_value()
        return self._value


def print_to_string(target='STDOUT'):
    return _PrintToStringContext(target, need_lock=True)

