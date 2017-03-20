# -*- coding:utf8 -*-
# File   : cnn.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/30/16
#
# This file is part of TensorArtist

from ._defaults import __default_dtype__, __default_nonlin__
from ._migrate import zeros, ones
from .helper import as_varnode, get_4dshape, get_2dshape, wrap_varnode_func, wrap_named_op, StaticDynamicDim
from .shape import flatten2, remove_axis, canonize_sym_shape
from .netsrc import variable, ensure_variable
from ..graph.env import Env, get_default_env

import tensorflow as tf
import functools

__all__ = ['conv2d', 'pooling2d', 'max_pooling2d', 'avg_pooling2d', 'fc', 'dropout', 'batch_norm', 'deconv2d']

@wrap_named_op
@wrap_varnode_func
def conv2d(name, inpvar, nr_output_channels, kernel, stride=1, padding='SAME',
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
    if W is None:
        W = tf.contrib.layers.xavier_initializer_conv2d()
    W = ensure_variable('W', W, shape=W_shape, dtype=param_dtype)

    if use_bias:
        if bias_is_shared_in_channel:
            b_shape = (cout, )
        else:
            assert inpvar.static_shape[1] is not None and inpvar.static_shape[2] is not None
            b_shape = inpvar.static_shape[1:3] + (cout, )

        if b is None:
            b = tf.constant_initializer()
        b = ensure_variable('b', b, shape=b_shape, dtype=param_dtype)

    _ = inpvar
    _ = tf.nn.conv2d(_, W, strides=stride, padding=padding, name='conv')
    if use_bias:
        _ = tf.nn.bias_add(_, b, name='bias')
    _ = nonlin(_, name='nonlin')

    return tf.identity(_, name='out')


@wrap_named_op
@wrap_varnode_func
def pooling2d(name, inpvar, kernel, stride=None, padding='VALID', method='MAX'):
    inpvar = as_varnode(inpvar)
    kernel = get_4dshape(kernel)
    stride = get_4dshape(stride, kernel)

    assert inpvar.ndims == 4

    if method == 'MAX':
        func = tf.nn.max_pool
    else:
        assert method == 'AVG'
        func = tf.nn.avg_pool

    return func(inpvar, ksize=kernel, strides=stride, padding=padding, name='out')


max_pooling2d = functools.partial(pooling2d, method='MAX')
avg_pooling2d = functools.partial(pooling2d, method='AVG')


@wrap_named_op
@wrap_varnode_func
def fc(name, inpvar, nr_output_channels,
        use_bias=True, nonlin=__default_nonlin__,
        W=None, b=None, param_dtype=__default_dtype__):
    inpvar = flatten2(inpvar)

    assert inpvar.static_shape[1] is not None
    W_shape = (inpvar.static_shape[1], nr_output_channels)
    b_shape = (nr_output_channels, )

    if W is None:
        W = tf.contrib.layers.xavier_initializer_conv2d()
    W = ensure_variable('W', W, shape=W_shape, dtype=param_dtype)
    if use_bias:
        if b is None:
            b = tf.constant_initializer()
        b = ensure_variable('b', b, shape=b_shape, dtype=param_dtype)

    out = tf.nn.xw_plus_b(inpvar, W, b, name='xwpb') if use_bias else tf.matmul(inpvar, W, name='matmul')
    out = nonlin(out, name='nonlin')
    return tf.identity(out, name='out')


@wrap_named_op
@wrap_varnode_func
def dropout(name, inpvar, keep_prob, keep_prob_sym=None, noise_shape=None, seed=None):
    env = get_default_env()
    if env.flags.compute_enable_dropout(name):
        keep_prob_sym = keep_prob
        out = tf.nn.dropout(inpvar, keep_prob_sym, noise_shape=noise_shape, seed=seed, name='dropout')
    else:
        out = inpvar
    return tf.identity(out, name='out')


@wrap_named_op
@wrap_varnode_func
def batch_norm(name, inpvar, decay=0.9, epsilon=1e-5, use_affine=True, param_dtype=__default_dtype__):
    ''' inpvar should be of data_format NHWC'''
    from tensorflow.python.training import moving_averages
    assign_moving_average = moving_averages.assign_moving_average

    inpvar = as_varnode(inpvar)
    shape = inpvar.static_shape

    assert len(shape) in [2, 4]
    nr_channels = shape[-1]
    if len(shape) == 2:
        inpvar = inpvar.reshape(-1, 1, 1, nr_channels)

    if use_affine:
        beta = variable('beta', tf.constant_initializer(), shape=[nr_channels], dtype=param_dtype)
        gamma = variable('gamma', tf.constant_initializer(1.0), shape=[nr_channels], dtype=param_dtype)
    else:
        beta = zeros([nr_channels], name='beta')
        gamma = ones([nr_channels], name='gamma')
    moving_mean = variable('mean/ema', tf.constant_initializer(), shape=[nr_channels], trainable=False)
    moving_var = variable('variance/ema', tf.constant_initializer(), shape=[nr_channels], trainable=False)

    env = get_default_env()
    if env.flags.compute_update_batch_normalization(name):
        xn, batch_mean, batch_var = tf.nn.fused_batch_norm(inpvar, gamma, beta, epsilon=epsilon, is_training=True, name='bn')
    else:
        xn = tf.nn.batch_normalization(inpvar, moving_mean, moving_var, beta, gamma, variance_epsilon=epsilon, name='bn')

    if len(shape) == 2:
        xn = remove_axis(xn, [1, 2])

    if env.flags.compute_update_batch_normalization(name) and env.current_dpc.is_master_device:
        update_mean_op = assign_moving_average(moving_mean.impl, batch_mean, decay, zero_debias=False, name='mean_ema_op')
        update_var_op = assign_moving_average(moving_var.impl, batch_var, decay, zero_debias=False, name='var_ema_op')

        with tf.control_dependencies([update_mean_op, update_var_op]):
            return tf.identity(xn, name='out')
    else:
        return tf.identity(xn, name='out')


@wrap_named_op
@wrap_varnode_func
def deconv2d(name, inpvar, nr_output_channels, kernel, stride=1, padding='SAME',
        use_bias=True, bias_is_shared_in_channel=True,
        nonlin=__default_nonlin__,
        W=None, b=None, param_dtype=__default_dtype__):

    inpvar = as_varnode(inpvar)
    in_shape = inpvar.static_shape
    nr_input_channels = in_shape[3]
    assert nr_input_channels is not None

    kernel = get_2dshape(kernel)
    stride2 = get_2dshape(stride)
    stride4 = get_4dshape(stride)

    sd_h = StaticDynamicDim(in_shape[1], inpvar.shape[1]) * stride2[0]
    sd_w = StaticDynamicDim(in_shape[2], inpvar.shape[2]) * stride2[1]
    out_shape_static = [in_shape[0], sd_h.static, sd_w.static, nr_output_channels]
    out_shape_dynamic = canonize_sym_shape([inpvar.shape[0], sd_h.dynamic, sd_w.dynamic, nr_output_channels])

    W_shape = kernel + (nr_output_channels, nr_input_channels)

    if W is None:
        W = tf.contrib.layers.xavier_initializer_conv2d()
    W = ensure_variable('W', W, shape=W_shape, dtype=param_dtype)
    if use_bias:
        if bias_is_shared_in_channel:
            b_shape = (nr_output_channels, )
        else:
            assert in_shape[1] is not None and in_shape[2] is not None
            b_shape = in_shape[1:3] + (nr_output_channels, )

        if b is None:
            b = tf.constant_initializer()
        b = ensure_variable('b', b, shape=b_shape, dtype=param_dtype)

    _ = inpvar
    _ = tf.nn.conv2d_transpose(_, W, out_shape_dynamic, stride4, padding=padding, data_format='NHWC', name='conv')
    _.set_shape(tf.TensorShape(out_shape_static))
    if use_bias:
        _ = tf.nn.bias_add(_, b, name='bias')
    _ = nonlin(_, name='nonlin')

    return tf.identity(_, name='out')

