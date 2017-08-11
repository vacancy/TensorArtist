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

__all__ = ['stprint', 'stformat']

__stprint_locks = collections.defaultdict(threading.Lock)


def _indent_print(msg, indent, prefix=None, end='\n', file=sys.stdout):
    print(*['  '] * indent, end='', file=file)
    if prefix is not None:
        print(prefix, end='', file=file)
    print(msg, end=end, file=file)


def stprint(data, key=None, indent=0, file=sys.stdout, need_lock=True):
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
