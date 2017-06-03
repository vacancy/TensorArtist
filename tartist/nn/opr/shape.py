# -*- coding:utf8 -*-
# File   : shape.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/17/17
# 
# This file is part of TensorArtist.


from ._defaults import __default_dtype__, __default_nonlin__
from .helper import as_varnode, wrap_varnode_func, wrap_simple_named_op
import tensorflow as tf
import numpy as np

__all__ = [
    'canonize_sym_shape',
    'flatten', 'flatten2', 
    'reshape', 'dimshuffle', 'transpose', 'broadcast', 'tile',
    'add_axis', 'expand_dim', 'remove_axis', 'sqeeze'
]


@wrap_simple_named_op
def canonize_sym_shape(shape, name='canonize_shape'):
    if type(shape) in (tuple, list) and type(shape[0]) in (tuple, list):
        shape = shape[0] 
    
    if not isinstance(shape, (tuple, list)):
        return shape
    if all(map(lambda x: type(x) is int, shape)):
        return shape
    return tf.stack(shape)


@wrap_simple_named_op
@wrap_varnode_func
def flatten(inpvar, name='flatten'):
    return tf.reshape(inpvar, [-1])


@wrap_simple_named_op
@wrap_varnode_func
def flatten2(inpvar, name='flatten2'):
    inpvar = as_varnode(inpvar)
    shape = inpvar.static_shape[1:]

    if None not in shape:
        return tf.reshape(inpvar, [-1, np.prod(shape)])
    out = tf.reshape(inpvar, tf.stack([tf.shape(inpvar)[0], -1]))
    return tf.identity(out, name='out')


@wrap_simple_named_op
@wrap_varnode_func
def reshape(inpvar, tshape, name='reshape'):
    return tf.reshape(inpvar, canonize_sym_shape(tshape), name='out')


@wrap_simple_named_op
@wrap_varnode_func
def dimshuffle(inpvar, perm, name='dimshuffle'):
    return tf.transpose(inpvar, canonize_sym_shape(perm), name='out')

transpose = dimshuffle


@wrap_varnode_func
def add_axis(inpvar, axis, name='add_axis'):
    return tf.expand_dims(inpvar, axis=axis, name=name)

expand_dim = add_axis


@wrap_varnode_func
def remove_axis(inpvar, axis, name='remove_axis'):
    return tf.squeeze(inpvar, axis=axis, name=name)

sqeeze = remove_axis


@wrap_simple_named_op
@wrap_varnode_func
def tile(inpvar, multiples, name='tile'):
    return tf.tile(inpvar, canonize_sym_shape(multiples), name=name)


@wrap_simple_named_op
@wrap_varnode_func
def broadcast(inpvar, tshape, name='broadcast'):
    sshape = tf.shape(inpvar)
    tshape = canonize_sym_shape(tshape)
    multiples = tshape // sshape 
    return tf.tile(inpvar, multiples, name='out')
