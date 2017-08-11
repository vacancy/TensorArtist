# -*- coding:utf8 -*-
# File   : thirdparty.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 4/28/17
# 
# This file is part of TensorArtist.


from IPython import embed
from tqdm import tqdm as _tqdm
from tartist.core.utils.thirdparty import get_tqdm_defaults


def tqdm(iterable, **kwargs):
    """Wrapped tqdm, where default kwargs will be load, and support `for i in tqdm(10)` usage.
    """
    for k, v in get_tqdm_defaults().items():
        kwargs.setdefault(k, v)

    if type(iterable) is int:
        iterable, total = range(iterable), iterable
    else:
        try:
            total = len(iterable)
        except:
            total = None

    if 'total' not in kwargs and total is not None:
        kwargs['total'] = total

    return _tqdm(iterable, **kwargs)

__all__ = ['embed', 'tqdm']
