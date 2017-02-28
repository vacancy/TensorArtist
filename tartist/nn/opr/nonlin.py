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
from .cnn import batchnorm
from ._migrate import relu

__all__ = ['bn_relu']

@wrap_varnode_func
def bn_relu(inpvar, name=None):
    _ = batchnorm('bn', inpvar)
    _ = relu(_, name=name)
    return _

