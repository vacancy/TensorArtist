# -*- coding:utf8 -*-
# File   : _migrate.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/24/17
#
# This file is part of TensorArtist.

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


# unary arith
identity = migrate_opr('identity', tf.identity)

neg = migrate_opr('neg', tf.negative)
abs = migrate_opr('abs', tf.abs)
log = migrate_opr('log', tf.log)
exp = migrate_opr('exp', tf.exp)
reciprocal = migrate_opr('reciprocal', tf.reciprocal)
sqr = migrate_opr('sqr', tf.square)
sqrt = migrate_opr('sqrt', tf.sqrt)
rsqrt = migrate_opr('rsqrt', tf.rsqrt)
floor = migrate_opr('floor', tf.floor)
ceil = migrate_opr('ceil', tf.ceil)
round = migrate_opr('round', tf.round)
sin = migrate_opr('sin', tf.sin)
cos = migrate_opr('cos', tf.cos)
tan = migrate_opr('tan', tf.tan)
asin = migrate_opr('asin', tf.asin)
acos = migrate_opr('acos', tf.acos)
atan = migrate_opr('atan', tf.atan)
tanh = migrate_opr('tanh', tf.tanh)

# unary arith: advanced
sigmoid = migrate_opr('sigmoid', tf.sigmoid)
relu = migrate_opr('relu', tf.nn.relu)

# binary arith
add = migrate_opr('add', tf.add)
sub = migrate_opr('sub', tf.subtract)
mul = migrate_opr('mul', tf.multiply)
truediv = migrate_opr('truediv', tf.truediv)
floordiv = migrate_opr('floordiv', tf.floordiv)
mod = migrate_opr('mod', tf.mod)
cross = migrate_opr('cross', tf.cross)
matmul = migrate_opr('matmul', tf.matmul)
bias_add = migrate_opr('bias_add', tf.nn.bias_add)
max = migrate_opr('max', tf.maximum)
min = migrate_opr('min', tf.minimum)
argmax = migrate_opr('argmax', tf.maximum)
argmin = migrate_opr('argmin', tf.minimum)
pow = migrate_opr('pow', tf.pow)

eq = migrate_opr('eq', tf.equal)
neq = migrate_opr('neq', tf.not_equal)
gt = migrate_opr('gt', tf.greater)
ge = migrate_opr('ge', tf.greater_equal)
lt = migrate_opr('lt', tf.less)
le = migrate_opr('le', tf.less_equal)

add_n = migrate_opr('add_n', tf.add_n)

reduce_sum = migrate_opr('reduce_sum', tf.reduce_sum)
reduce_prod = migrate_opr('reduce_prod', tf.reduce_prod)
reduce_mean = migrate_opr('reduce_mean', tf.reduce_mean)
reduce_max = migrate_opr('reduce_max', tf.reduce_max)
reduce_min = migrate_opr('reduce_min', tf.reduce_min)
reduce_all = migrate_opr('reduce_all', tf.reduce_all)
reduce_any = migrate_opr('reduce_any', tf.reduce_any)

slice = migrate_opr('slice', tf.slice)
strided_slice = migrate_opr('strided_slice', tf.strided_slice)
split = migrate_opr('split', tf.split)
pad = migrate_opr('pad', tf.pad)
reverse = migrate_opr('reverse', tf.reverse)

# array ops: creation
cast = migrate_opr('cast', tf.cast)
zeros = migrate_opr('zeros', tf.zeros)
ones = migrate_opr('ones', tf.ones)
zeros_like = migrate_opr('zeros_like', tf.zeros_like)
ones_like = migrate_opr('ones_like', tf.ones_like)
one_hot = migrate_opr('one_hot', tf.one_hot)
range = migrate_opr('range', tf.range)

# array ops: advanced
where = migrate_opr('where', tf.where)
meshgrid = migrate_opr('meshgrid', tf.meshgrid)

# condition, loop
cond = migrate_opr('cond', tf.cond)

# control
zero_grad = migrate_opr('zero_grad', tf.stop_gradient)

# softmax related
softmax = migrate_opr('softmax', tf.nn.softmax)
softmax_cross_entropy_with_logits = migrate_opr('softmax_cross_entropy_with_logits', tf.nn.softmax_cross_entropy_with_logits)
sparse_softmax_cross_entropy_with_logits = migrate_opr('sparse_softmax_cross_entropy_with_logits', tf.nn.sparse_softmax_cross_entropy_with_logits)
sigmoid_cross_entropy_with_logits = migrate_opr('sigmoid_cross_entropy_with_logits', tf.nn.sigmoid_cross_entropy_with_logits)

# clipping
clip_by_value = migrate_opr('clip_by_value', tf.clip_by_value)
clip_by_norm = migrate_opr('clip_by_norm', tf.clip_by_norm)
clip_by_global_norm = migrate_opr('clip_by_global_norm', tf.clip_by_global_norm)
clip_by_average_norm = migrate_opr('clip_by_average_norm', tf.clip_by_average_norm)

del functools
del tf
