# -*- coding:utf8 -*-
# File   : shape.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/17/17
# 
# This file is part of TensorArtist.


from .helper import as_varnode, wrap_named_op
import tensorflow as tf
import numpy as np

__all__ = [
    'canonize_sym_shape',
    'flatten', 'flatten2', 
    'reshape', 'dimshuffle', 'transpose', 'broadcast', 'tile',
    'add_axis', 'expand_dim', 'remove_axis', 'sqeeze'
]


@wrap_named_op
def canonize_sym_shape(shape, name='canonize_shape'):
    if type(shape) in (tuple, list) and type(shape[0]) in (tuple, list):
        shape = shape[0] 
    
    if not isinstance(shape, (tuple, list)):
        return shape
    if all(map(lambda x: type(x) is int, shape)):
        return np.array(shape, dtype='int32')
    return tf.stack(shape)


@wrap_named_op
def flatten(inpvar, name='flatten'):
    return tf.reshape(inpvar, [-1])


@wrap_named_op
def flatten2(inpvar, name='flatten2'):
    inpvar = as_varnode(inpvar)
    shape = inpvar.static_shape[1:]

    if None not in shape:
        return tf.reshape(inpvar, [-1, np.prod(shape)])
    out = tf.reshape(inpvar, tf.stack([tf.shape(inpvar)[0], -1]))
    return tf.identity(out, name='out')


@wrap_named_op
def reshape(inpvar, tshape, name='reshape'):
    return tf.reshape(inpvar, canonize_sym_shape(tshape), name='out')


@wrap_named_op
def dimshuffle(inpvar, perm, name='dimshuffle'):
    return tf.transpose(inpvar, canonize_sym_shape(perm), name='out')

transpose = dimshuffle


@wrap_named_op(use_scope=False)
def add_axis(inpvar, axis, name='add_axis'):
    return tf.expand_dims(inpvar, axis=axis, name=name)

expand_dim = add_axis


@wrap_named_op(use_scope=False)
def remove_axis(inpvar, axis, name='remove_axis'):
    return tf.squeeze(inpvar, axis=axis, name=name)

sqeeze = remove_axis


@wrap_named_op
def tile(inpvar, multiples, name='tile'):
    return tf.tile(inpvar, canonize_sym_shape(multiples), name=name)


@wrap_named_op
def broadcast(inpvar, tshape, name='broadcast'):
    sshape = tf.shape(inpvar)
    tshape = canonize_sym_shape(tshape)
    multiples = tshape // sshape 
    return tf.tile(inpvar, multiples, name='out')
