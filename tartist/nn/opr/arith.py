# -*- coding:utf8 -*-
# File   : aris.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/27/17
# 
# This file is part of TensorArtist

from .helper import as_varnode, wrap_varnode_func, wrap_simple_named_op
from .helper import lazy_O as O

__all__ = ['rms', 'std']


@wrap_simple_named_op
@wrap_varnode_func
def rms(inpvar, name='rms'):
    return O.sqrt((as_varnode(inpvar) ** 2.).mean(), name=name)


@wrap_simple_named_op
@wrap_varnode_func
def std(inpvar, name='std'):
    inpvar = as_varnode(inpvar)
    return O.sqrt(((inpvar - inpvar.mean()) ** 2.).mean(), name=name)

