# -*- coding:utf8 -*-
# File   : cnn.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/30/16
# 
# This file is part of TensorArtist

from ._defaults import __default_dtype__, __default_nonlin__
from .helper import as_varnode, get_4dshape, get_2dshape, wrap_varnode_func, wrap_named_op
from .shape import flatten2
from .netsrc import variable
import tensorflow as tf

__all__ = ['conv2d', 'pooling2d', 'fc']


@wrap_named_op
@wrap_varnode_func
def conv2d(name, inpvar, nr_output_channels, kernel, stride=1, padding='VALID',
        use_bias=True, bias_is_shared_in_channel=True,
        nonlin=__default_nonlin__,
        W=None, b=None, param_dtype=__default_dtype__):

    inpvar = as_varnode(inpvar)
    kernel = get_2dshape(kernel)
    stride = get_4dshape(stride)

    assert inpvar.ndims == 4
    assert padding in ('VALID', 'SAME')

    assert inpvar.static_shape[3] is not None
    cin, cout = inpvar.static_shape[3], nr_output_channels
    W_shape = kernel + (cin, cout)
    if use_bias:
        if bias_is_shared_in_channel:
            b_shape = (cout, )
        else:
            assert inpvar.static_shape[1] is not None and inpvar.static_shape[2] is not None
            b_shape = inpvar.static_shape[1:3] + (cout, )

    with tf.variable_scope(name, reuse=False):
        if W is None:
            W = variable('W', tf.contrib.layers.xavier_initializer_conv2d(), shape=W_shape, dtype=param_dtype)
        if use_bias:
            if b is None:
                b = variable('b', tf.constant_initializer(), shape=b_shape, dtype=param_dtype)

    _ = inpvar.impl
    _ = tf.nn.conv2d(_, W, strides=stride, padding=padding, name=name)
    if use_bias:
        _ = tf.nn.bias_add(_, b, name=name + '_bias')
    _ = nonlin(_)
    return _


@wrap_named_op
@wrap_varnode_func
def pooling2d(name, inpvar, kernel, stride=None, padding='VALID', method='MAX'):
    inpvar = as_varnode(inpvar)
    kernel = get_4dshape(kernel)
    stride = get_4dshape(stride, kernel)

    assert inpvar.ndims == 4
    assert method == 'MAX'

    return tf.nn.max_pool(inpvar, ksize=kernel, strides=stride, padding=padding, name=name)


@wrap_named_op
@wrap_varnode_func
def fc(name, inpvar, nr_output_channels,
        use_bias=True, nonlin=__default_nonlin__,
        W=None, b=None, param_dtype=__default_dtype__):

    inpvar = flatten2(inpvar)

    assert inpvar.static_shape[1] is not None
    W_shape = (inpvar.static_shape[1], nr_output_channels)
    b_shape = (nr_output_channels, )

    with tf.variable_scope(name, reuse=False):
        if W is None:
            W = variable('W', tf.contrib.layers.xavier_initializer_conv2d(), shape=W_shape, dtype=param_dtype)
        if use_bias:
            if b is None:
                b = variable('b', tf.constant_initializer(), shape=b_shape, dtype=param_dtype)

    out = tf.nn.xw_plus_b(inpvar, W, b) if use_bias else tf.matmul(inpvar, W)
    out = nonlin(out)
    return out

