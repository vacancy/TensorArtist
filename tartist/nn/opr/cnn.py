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
from ..graph.env import Env, get_default_env

import tensorflow as tf
from tensorflow.python.training import moving_averages

__all__ = ['conv2d', 'pooling2d', 'fc', 'dropout', 'batchnorm']


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


@wrap_named_op
@wrap_varnode_func
def dropout(name, inpvar, keep_prob, keep_prob_sym=None, noise_shape=None, seed=None):
    if keep_prob_sym is None:
        env = get_default_env()
        keep_prob_sym = keep_prob if env.phase == Env.Phase.TRAIN else 1
    out = tf.nn.dropout(inpvar, keep_prob_sym, noise_shape=noise_shape, seed=seed, name=name)
    return out


@wrap_named_op
@wrap_varnode_func
def batchnorm(name, inpvar, use_local_stat=None, decay=0.9, epsilon=1e-5, use_affine=True, param_dtype=__default_dtype__):
    '''
    inpvar(tf.Tensor): NHWC
    '''
    inpvar = as_varnode(inpvar)
    shape = inpvar.static_shape
    assert len(shape) in [2, 4]
    out = shape[-1]
    if len(shape) == 2:
        inpvar = inpvar.reshape(-1, 1, 1, out)
    with tf.variable_scope(name):
        if use_affine:
            beta = variable('beta', tf.constant_initializer(), shape=[out], dtype=param_dtype)
            gamma = variable('gamma', tf.constant_initializer(1.0), shape=[out], dtype=param_dtype)
        else:
            beta = tf.zeros([out], name='beta')
            gamma = tf.ones([out], name='gamma')
        moving_mean = variable('mean/EMA', tf.constant_initializer(), shape=[out], trainable=False)
        moving_var = variable('variance/EMA', tf.constant_initializer(), shape=[out], trainable=False)
    env = get_default_env()
    if env.phase == Env.Phase.TRAIN:
        xn, batch_mean, batch_var =  tf.nn.fused_batch_norm(inpvar, gamma, beta, epsilon=epsilon, is_training=True)
    else:
        xn, _, _ = tf.nn.fused_batch_norm(inpvar, gamma, beta, moving_mean, moving_var, epsilon=epsilon, is_training=False)
    if len(shape) == 2:
        xn = tf.squeeze(xn, [1, 2])

    if env.current_dpc.is_master_device:
        update_mean_op = moving_averages.assign_moving_average(moving_mean.impl, batch_mean, decay, zero_debias=False, name='mean_ema_op')
        update_var_op = moving_averages.assign_moving_average(moving_var.impl, batch_var, decay, zero_debias=False, name='var_ema_op')
        with tf.control_dependencies([update_mean_op, update_var_op]):
            return tf.identity(xn, name=name)
    else:
        return tf.identity(xn, name=name)

