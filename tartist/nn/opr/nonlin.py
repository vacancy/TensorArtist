# -*- coding:utf8 -*-
# File   : nonlin.py
# Author : Jiayuan Mao
#          Honghua Dong
# Email  : maojiayuan@gmail.com
#          dhh19951@gmail.com
# Date   : 2/27/17
#
# This file is part of TensorArtist

import tensorflow as tf

from .helper import as_varnode, get_4dshape, get_2dshape, wrap_varnode_func, wrap_named_op
from .cnn import batch_norm
from ._migrate import max, relu

__all__ = ['leaky_relu', 'bn_relu', 'bn_nonlin']


@wrap_varnode_func
def leaky_relu(x, alpha, name='output'):
    return max(x, alpha * x, name=name)


@wrap_varnode_func
def bn_relu(inpvar, name=None):
    _ = batch_norm('bn', inpvar)
    _ = relu(_, name='relu')
    return _


@wrap_varnode_func
def bn_nonlin(inpvar, name=None):
    _ = batch_norm('bn', inpvar)
    return _

