# -*- coding:utf8 -*-
# File   : print.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 5/4/17
# 
# This file is part of TensorArtist.

import numpy as np

__all__ = ['stprint']


def _indent_print(msg, indent, prefix=None, end='\n'):
    print(*['  '] * indent, end='')
    if prefix is not None:
        print(prefix, end='')
    print(msg, end=end)


def stprint(data, key=None, indent=0):
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

    if t is tuple:
        _indent_print('tuple[', indent, prefix=key)
        for v in data:
            stprint(v, indent=indent + 1)
        _indent_print(']', indent)
    elif t is list:
        _indent_print('list[', indent, prefix=key)
        for v in data:
            stprint(v, indent=indent + 1)
        _indent_print(']', indent)
    elif t is dict:
        _indent_print('dict{', indent, prefix=key)
        for k in sorted(data.keys()):
            v = data[k]
            stprint(v, indent=indent + 1, key='{}: '.format(k))
        _indent_print('}', indent)
    elif t is np.ndarray:
        _indent_print('ndarray{}, dtype={}'.format(data.shape, data.dtype), indent, prefix=key)
    else:
        _indent_print(data, indent, prefix=key)
