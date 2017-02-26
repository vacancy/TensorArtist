# -*- coding:utf8 -*-
# File   : aris.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/27/17
# 
# This file is part of TensorArtist

from .helper import as_varnode, wrap_varnode_func, wrap_named_op
import tensorflow as tf

__all__ = ['rms', 'std']


def rms(inpvar, name=None):
    from ._migrate import sqrt
    return sqrt((as_varnode(inpvar) ** 2).mean(), name=name)

def std(inpvar, name=None):
    from ._migrate import sqrt
    inpvar = as_varnode(inpvar)
    return sqrt(((inpvar - inpvar.mean()) ** 2).mean(), name=name)

