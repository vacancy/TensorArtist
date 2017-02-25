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

__all__ = [
    'flatten', 'flatten2', 
    'reshape', 'dimshuffle', 'broadcast', 'tile', 
    'add_axis', 'remove_axis'
]


def canonize_sym_shape(shape):
    if type(shape) in (tuple, list) and type(shape[0]) in (tuple, list):
        shape = shape[0] 

    if all(map(lambda x: type(x) is int, shape)):
        return shape
    return tf.stack(shape)


@wrap_varnode_func
def flatten(inpvar):
    return tf.reshape(inpvar, [-1])


@wrap_varnode_func
def flatten2(inpvar):
    inpvar = as_varnode(inpvar)
    shape = inpvar.static_shape[1:]

    if None not in shape:
        return tf.reshape(inpvar, [-1, np.prod(shape)])
    return tf.reshape(inpvar, tf.stack([tf.shape(inpvar)[0], -1]))


@wrap_varnode_func
def reshape(inpvar, tshape, name=None):
    return tf.reshape(inpvar, canonize_sym_shape(tshape), name=name)


@wrap_varnode_func
def dimshuffle(inpvar, perm, name=None):
    return tf.transpose(inpvar, canonize_sym_shape(perm), name=name)


@wrap_varnode_func
def add_axis(inpvar, axis, name=None):
    return tf.expand_dims(inpvar, axis=axis, name=name)


@wrap_varnode_func
def remove_axis(inpvar, axis, name=None):
    return tf.squeeze(inpvar, axis=axis, name=name)


@wrap_varnode_func
def sqeeze(inpvar, axis=None, name=None):
    return tf.squeeze(inpvar, axis=axis, name=name)


@wrap_varnode_func
def tile(inpvar, multiples, name=None):
    return tf.tile(inpvar, canonize_sym_shape(multiples), name=name)


@wrap_varnode_func
def broadcast(inpvar, tshape, name=None):
    sshape = tf.shape(inpvar)
    tshape = canonize_sym_shape(tshape)
    multiples = tshape // sshape 
    return tf.tile(inpvar, multiples, name=name)

