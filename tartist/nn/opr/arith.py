# -*- coding:utf8 -*-
# File   : arith.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/27/17
# 
# This file is part of TensorArtist.

from ..graph import get_default_env
from .helper import as_varnode, wrap_named_op
from .helper import lazy_O as O

import tensorflow as tf

__all__ = ['rms', 'std', 'atanh', 'logit', 'moving_average']


@wrap_named_op
def rms(inpvar, name='rms'):
    return O.sqrt((as_varnode(inpvar) ** 2.).mean(), name='out')


@wrap_named_op
def std(inpvar, name='std'):
    inpvar = as_varnode(inpvar)
    return O.sqrt(((inpvar - inpvar.mean()) ** 2.).mean(), name='out')


@wrap_named_op
def atanh(inpvar, name='atanh', eps=1e-6):
    inpvar = as_varnode(inpvar)
    return O.identity(0.5 * O.log((1. + inpvar) / (1. - inpvar + eps) + eps), name='out')


@wrap_named_op
def logit(inpvar, name='logit', eps=1e-6):
    inpvar = as_varnode(inpvar)
    return O.identity(0.5 * O.log(inpvar / (1. - inpvar + eps) + eps), name='out')


@wrap_named_op(use_variable_scope=True)
def moving_average(inpvar, name='moving_average', eps=1e-6):
    inpvar = O.as_varnode(inpvar)

    target_shape = inpvar.static_shape[1:]
    assert all(map(lambda x: x is not None, target_shape)), 'Input shape must be known.'

    mu = O.variable('mu', tf.constant_initializer(0), shape=target_shape, trainable=False)
    var = O.variable('var', tf.constant_initializer(0), shape=target_shape, trainable=False)
    n = O.variable('n', tf.constant_initializer(0), shape=(1, ), trainable=False)

    batch_mu = inpvar.mean(axis=0)
    batch_var = ((inpvar - batch_mu.add_axis(0)) ** 2).mean(axis=0)

    mu_update = tf.assign(mu.tft, mu * n / (n + 1.) + batch_mu / (n + 1.))
    var_update = tf.assign(var.tft, var * n / (n + 1.) + batch_var / (n + 1.))
    n_update = tf.assign(n.tft, n + 1.)
    updates = tf.group(mu_update, var_update, n_update)

    env = get_default_env()
    if env.flags.compute_update_batch_normalization(name):
        with tf.control_dependencies([updates]):
            std = O.sqrt(var + eps)
            return (inpvar - mu) / std
    else:
        std = O.sqrt(var + eps)
        return (inpvar - mu) / std
