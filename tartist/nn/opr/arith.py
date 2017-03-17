# -*- coding:utf8 -*-
# File   : aris.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/27/17
# 
# This file is part of TensorArtist

from .helper import as_varnode, wrap_varnode_func

__all__ = ['rms', 'std']


@wrap_varnode_func
def rms(inpvar, name=None):
    from ._migrate import sqrt
    return sqrt((as_varnode(inpvar) ** 2.).mean(), name=name)


@wrap_varnode_func
def std(inpvar, name=None):
    from ._migrate import sqrt
    inpvar = as_varnode(inpvar)
    return sqrt(((inpvar - inpvar.mean()) ** 2.).mean(), name=name)

