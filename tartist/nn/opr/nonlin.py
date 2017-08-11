# -*- coding:utf8 -*-
# File   : nonlin.py
# Author : Jiayuan Mao
#          Honghua Dong
# Email  : maojiayuan@gmail.com
#          dhh19951@gmail.com
# Date   : 2/27/17
#
# This file is part of TensorArtist.

import functools
import tensorflow as tf

from . import migrate as migrate_oprs
from .helper import wrap_named_op
from .helper import lazy_O as O

__all__ = [
        'bn_nonlin', 'wrap_bn_nonlin',
        'p_relu', 'leaky_relu', 'p_elu', 'selu',
        'abs_tanh', 
        'log_abs', 'log_absp1',
        'xlogx',
        'bn_relu', 'bn_tanh', 'bn_leaky_relu'
]


@wrap_named_op(use_variable_scope=True)
def bn_nonlin(inpvar, name='bn_nonlin'):
    _ = O.batch_norm('bn', inpvar)
    return O.identity(_, 'out')


def wrap_bn_nonlin(nonlin):
    default_name = 'bn_{}'.format(nonlin.__name__)

    @functools.wraps(nonlin)
    def new_nonlin(x, *args, name=default_name, **kwargs):
        # MJY(20170627): For p_relu backward compatibility. Use named_scope instead of variable_scope.
        with tf.variable_scope(name):
            _ = x
            _ = bn_nonlin(_)
            _ = nonlin(_, *args, **kwargs)
            _ = O.identity(_, name='out')
        return _
    new_nonlin.__name__ = default_name
    return new_nonlin


@wrap_named_op(use_variable_scope=True)
def p_relu(x, init=0.001, name='p_relu'):
    alpha = O.scalar('alpha', init)
    x = ((1. + alpha) * x + (1. - alpha) * abs(x))
    return O.mul(x, 0.5, name='out')


@wrap_named_op
def leaky_relu(x, alpha, name='leaky_relu'):
    # assert 0 <= alpha < 1
    if type(alpha) in (int, float):
        assert 0 <= alpha < 1
    return O.max(x, alpha * x, name='out')


@wrap_named_op
def p_elu(x, alpha, scale=1., name='p_elu'):
    return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))


@wrap_named_op
def selu(x, name='selu'):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))


@wrap_named_op
def abs_tanh(x, name='abs_tanh'):
    return O.abs(O.tanh(x))


@wrap_named_op
def log_abs(x, eps=1e-6, name='log_abs'):
    return O.log(O.abs(x) + eps)


@wrap_named_op
def log_absp1(x, name='log_abs'):
    return O.log(O.abs(x) + 1)


@wrap_named_op
def xlogx(x, eps=1e-6, name='xlogx'):
    return x * O.log(x + eps)


# For backward compatibility.
bn_relu = wrap_bn_nonlin(migrate_oprs.relu)
bn_tanh = wrap_bn_nonlin(migrate_oprs.tanh)
bn_leaky_relu = wrap_bn_nonlin(leaky_relu)
