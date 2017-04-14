# -*- coding:utf8 -*-
# File   : _migrate.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/24/17
# 
# This file is part of TensorArtist

from .helper import wrap_varnode_func

import tensorflow as tf
import functools


def migrate_opr(name, tf_func):
    @functools.wraps(tf_func)
    @wrap_varnode_func
    def new_func(*args, **kwargs):
        return tf_func(*args, **kwargs)
    new_func.__name__ = name

    return new_func


__all_migrated_oprs__ = []

# unary arith
__all_migrated_oprs__.extend([
    ('identity', tf.identity),

    ('neg', tf.negative),
    ('abs', tf.abs),
    ('log', tf.log),
    ('exp', tf.exp),
    ('reciprocal', tf.reciprocal),
    ('sqr', tf.square),
    ('sqrt', tf.sqrt),
    ('rsqrt', tf.rsqrt),
    ('floor', tf.floor),
    ('ceil', tf.ceil),
    ('round', tf.round),
    ('sin', tf.sin),
    ('cos', tf.cos),
    ('tan', tf.tan),
    ('asin', tf.asin),
    ('acos', tf.acos),
    ('atan', tf.atan),
    ('tanh', tf.tanh),
])

# unary arith: advanced
__all_migrated_oprs__.extend([
    ('sigmoid', tf.sigmoid),
    ('relu', tf.nn.relu),
])

# binary arith
__all_migrated_oprs__.extend([
    ('add', tf.add),
    ('sub', tf.subtract),
    ('mul', tf.multiply),
    ('truediv', tf.truediv),
    ('floordiv', tf.floordiv),
    ('mod', tf.mod),
    ('cross', tf.cross),
    ('matmul', tf.matmul),
    ('bias_add', tf.nn.bias_add),
    ('max', tf.maximum),
    ('min', tf.minimum),
    ('argmax', tf.maximum),
    ('argmin', tf.minimum),
    ('pow', tf.pow),

    ('eq', tf.equal),
    ('neq', tf.not_equal),
    ('gt', tf.greater),
    ('ge', tf.greater_equal),
    ('lt', tf.less),
    ('le', tf.less_equal),

    ('add_n', tf.add_n)
])

__all_migrated_oprs__.extend([
    ('reduce_sum', tf.reduce_sum),
    ('reduce_prod', tf.reduce_prod),
    ('reduce_mean', tf.reduce_mean),
    ('reduce_max', tf.reduce_max),
    ('reduce_min', tf.reduce_min),
    ('reduce_all', tf.reduce_all),
    ('reduce_any', tf.reduce_any)
])

__all_migrated_oprs__.extend([
    ('slice', tf.slice),
    ('strided_slice', tf.strided_slice),
    ('split', tf.split),
    ('pad', tf.pad),
    ('reverse', tf.reverse),
])

# array ops: creation
__all_migrated_oprs__.extend([
    ('zeros', tf.zeros),
    ('ones', tf.ones),
    ('zeros_like', tf.zeros_like),
    ('ones_like', tf.ones_like),
    ('one_hot', tf.one_hot),
    ('range', tf.range)
])

# array ops: advanced
__all_migrated_oprs__.extend([
    ('where', tf.where),
    ('meshgrid', tf.meshgrid),
])

# condition, loop
__all_migrated_oprs__.extend([
    ('cond', tf.cond),
])

# control
__all_migrated_oprs__.extend([
    ('zero_grad', tf.stop_gradient)
])

# softmax related
__all_migrated_oprs__.extend([
    ('softmax', tf.nn.softmax),
    ('softmax_cross_entropy_with_logits', tf.nn.softmax_cross_entropy_with_logits),
    ('sparse_softmax_cross_entropy_with_logits', tf.nn.sparse_softmax_cross_entropy_with_logits),
    ('softmax_cross_entropy_with_logits', tf.nn.softmax),
    ('sigmoid_cross_entropy_with_logits', tf.nn.sigmoid_cross_entropy_with_logits),
])

# clipping
__all_migrated_oprs__.extend([
    ('clip_by_value', tf.clip_by_value),
    ('clip_by_norm', tf.clip_by_norm),
    ('clip_by_global_norm', tf.clip_by_global_norm),
    ('clip_by_average_norm', tf.clip_by_average_norm)
])

__all_migrated_oprs__ = dict(__all_migrated_oprs__)

__all__ = []
__all__.extend(list(__all_migrated_oprs__.keys()))

for k, v in __all_migrated_oprs__.items():
    globals()[k] = migrate_opr(k, v)

