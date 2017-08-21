# -*- coding:utf8 -*-
# File   : control.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 8/21/17
# 
# This file is part of TensorArtist.

from .helper import wrap_varnode_func
from ..graph.node import as_tftensor

import tensorflow as tf

__all__ = ['cond']


@wrap_varnode_func
def cond(pred, fn1, fn2, name=None):
    return tf.cond(pred, lambda: as_tftensor(fn1()), lambda: as_tftensor(fn2()), name=name)
