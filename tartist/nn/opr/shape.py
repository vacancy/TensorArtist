# -*- coding:utf8 -*-
# File   : shape.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/17/17
# 
# This file is part of TensorArtist


from ._defaults import __default_dtype__, __default_nonlin__
from .helper import as_varnode, wrap_varnode_func
import tensorflow as tf
import numpy as np

__all__ = ['flatten', 'flatten2']


def get_sym_shape(shape):
    if all(map(lambda x: type(x) is int, shape)):
        return shape
    return tf.pack(shape)


@wrap_varnode_func
def flatten(inpvar):
    return tf.reshape(inpvar, [-1])


@wrap_varnode_func
def flatten2(inpvar):
    inpvar = as_varnode(inpvar)
    shape = inpvar.static_shape[1:]

    if None not in shape:
        return tf.reshape(inpvar, [-1, np.prod(shape)])
    return tf.reshape(inpvar, tf.pack([tf.shape(inpvar)[0], -1]))


@wrap_varnode_func
def reshape(inpvar, shape, name=None):
    return tf.reshape(inpvar, get_sym_shape(shape), name=name)


@wrap_varnode_func
def dimshuffle(inpvar, perm, name=None):
    return tf.transpose(inpvar, get_sym_shape(perm), name=name)

