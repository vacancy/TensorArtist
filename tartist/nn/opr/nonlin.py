# -*- coding:utf8 -*-
# File   : nonlin.py
# Author : Jiayuan Mao
#          Honghua Dong
# Email  : maojiayuan@gmail.com
#          dhh19951@gmail.com
# Date   : 2/27/17
#
# This file is part of TensorArtist.

import tensorflow as tf

from .helper import wrap_named_op
from .helper import lazy_O as O

__all__ = ['p_relu', 'leaky_relu', 'bn_relu', 'bn_tanh', 'bn_leaky_relu', 'bn_nonlin', 'softplus']


@wrap_named_op
def p_relu(x, init=0.001, name='p_relu'):
    alpha = O.scalar('alpha', init)
    x = ((1. + alpha) * x + (1. - alpha) * abs(x))
    return O.mul(x, 0.5, name='out')


@wrap_named_op
def leaky_relu(x, alpha, name='leaky_relu'):
    return O.max(x, alpha * x, name='out')


@wrap_named_op
def bn_relu(inpvar, name='bn_relu'):
    _ = O.batch_norm('bn', inpvar)
    _ = O.relu(_, name='relu')
    return O.identity(_, name='out')


@wrap_named_op
def bn_tanh(inpvar, name='bn_tanh'):
    _ = O.batch_norm('bn', inpvar)
    _ = O.tanh(_, name='tanh')
    return O.identity(_, name='out')


@wrap_named_op
def bn_leaky_relu(inpvar, alpha, name='bn_leaky_relu'):
    _ = O.batch_norm('bn', inpvar)
    _ = O.leaky_relu(_, alpha, name='leaky_relu')
    return O.identity(_, name='out')


@wrap_named_op
def bn_nonlin(inpvar, name='bn_nonlin'):
    _ = O.batch_norm('bn', inpvar)
    return O.identity(_, 'out')


@wrap_named_op
def softplus(inpvar, name='softplus'):
    _ = O.log(1. + O.exp(inpvar))
    return O.identity(_, 'out')
