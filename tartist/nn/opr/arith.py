# -*- coding:utf8 -*-
# File   : aris.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/27/17
# 
# This file is part of TensorArtist.

from .helper import as_varnode, wrap_varnode_func, wrap_simple_named_op
from .helper import lazy_O as O

__all__ = ['rms', 'std', 'atanh', 'logit']


@wrap_simple_named_op
@wrap_varnode_func
def rms(inpvar, name='rms'):
    return O.sqrt((as_varnode(inpvar) ** 2.).mean(), name='out')


@wrap_simple_named_op
@wrap_varnode_func
def std(inpvar, name='std'):
    inpvar = as_varnode(inpvar)
    return O.sqrt(((inpvar - inpvar.mean()) ** 2.).mean(), name='out')


@wrap_simple_named_op
@wrap_varnode_func
def atanh(inpvar, name='atanh', eps=1e-6):
    inpvar = as_varnode(inpvar)
    return O.identity(0.5 * O.log((1. + inpvar) / (1. - inpvar + eps) + eps), name='out')


@wrap_simple_named_op
@wrap_varnode_func
def logit(inpvar, name='logit', eps=1e-6):
    inpvar = as_varnode(inpvar)
    return O.identity(0.5 * O.log(inpvar / (1. - inpvar + eps) + eps), name='out')
