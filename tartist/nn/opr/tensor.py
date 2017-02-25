# -*- coding:utf8 -*-
# File   : tensor.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/17/17
# 
# This file is part of TensorArtist


from ._defaults import __default_dtype__, __default_nonlin__
from .helper import as_varnode, wrap_varnode_func
import tensorflow as tf
import numpy as np

__all__ = ['concat', 'stack']


@wrap_varnode_func
def concat(inpvars, axis, name=None):
    return tf.concat(inpvars, axis=axis, name=name)


@wrap_varnode_func
def stack(inpvars, axis=0, name=None):
    return tf.stack(inpvars, axis=axis, name=name)

