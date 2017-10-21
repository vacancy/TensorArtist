# -*- coding:utf8 -*-
# File   : _migrate.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/24/17
#
# This file is part of TensorArtist.

from .helper import wrap_varnode_func as _wrap_varnode_func
from functools import wraps as _wraps
import tensorflow as _tf


def migrate_opr(name, tf_func):
    @_wraps(tf_func)
    @_wrap_varnode_func
    def new_func(*args, **kwargs):
        return tf_func(*args, **kwargs)
    new_func.__name__ = name

    return new_func


# unary arith
identity = migrate_opr('identity', _tf.identity)

neg = migrate_opr('neg', _tf.negative)
abs = migrate_opr('abs', _tf.abs)
log = migrate_opr('log', _tf.log)
log1p = migrate_opr('log1p', _tf.log1p)
exp = migrate_opr('exp', _tf.exp)
expm1 = migrate_opr('expm1', _tf.expm1)
reciprocal = migrate_opr('reciprocal', _tf.reciprocal)
sqr = migrate_opr('sqr', _tf.square)
sqrt = migrate_opr('sqrt', _tf.sqrt)
rsqrt = migrate_opr('rsqrt', _tf.rsqrt)
floor = migrate_opr('floor', _tf.floor)
ceil = migrate_opr('ceil', _tf.ceil)
round = migrate_opr('round', _tf.round)
sin = migrate_opr('sin', _tf.sin)
cos = migrate_opr('cos', _tf.cos)
tan = migrate_opr('tan', _tf.tan)
asin = migrate_opr('asin', _tf.asin)
acos = migrate_opr('acos', _tf.acos)
atan = migrate_opr('atan', _tf.atan)
tanh = migrate_opr('tanh', _tf.tanh)

# unary arith: advanced
sigmoid = migrate_opr('sigmoid', _tf.sigmoid)
relu = migrate_opr('relu', _tf.nn.relu)
relu6 = migrate_opr('relu6', _tf.nn.relu6)
crelu = migrate_opr('crelu', _tf.nn.crelu)
elu = migrate_opr('elu', _tf.nn.elu)
softplus = migrate_opr('softplus', _tf.nn.softplus)
softsign = migrate_opr('softsign', _tf.nn.softsign)
sgn = migrate_opr('sgn', _tf.sign)
lbeta = migrate_opr('lbeta', _tf.lbeta)
lgamma = migrate_opr('lgamma', _tf.lgamma)
digamma = migrate_opr('digamma', _tf.digamma)
erf = migrate_opr('erf', _tf.erf)
erfc = migrate_opr('erfc', _tf.erfc)

# binary/tenary arith: advanced
igamma = migrate_opr('igamma', _tf.igamma)
igammac = migrate_opr('igammac', _tf.igammac)
polygamma = migrate_opr('polygamma', _tf.polygamma)
zeta = migrate_opr('zeta', _tf.zeta)
betainc = migrate_opr('betainc', _tf.betainc)

# binary arith
add = migrate_opr('add', _tf.add)
sub = migrate_opr('sub', _tf.subtract)
mul = migrate_opr('mul', _tf.multiply)
truediv = migrate_opr('truediv', _tf.truediv)
floordiv = migrate_opr('floordiv', _tf.floordiv)
mod = migrate_opr('mod', _tf.mod)
cross = migrate_opr('cross', _tf.cross)
matmul = migrate_opr('matmul', _tf.matmul)
bias_add = migrate_opr('bias_add', _tf.nn.bias_add)
max = migrate_opr('max', _tf.maximum)
min = migrate_opr('min', _tf.minimum)
argmax = migrate_opr('argmax', _tf.argmax)
argmin = migrate_opr('argmin', _tf.argmin)
pow = migrate_opr('pow', _tf.pow)

eq = migrate_opr('eq', _tf.equal)
neq = migrate_opr('neq', _tf.not_equal)
gt = migrate_opr('gt', _tf.greater)
ge = migrate_opr('ge', _tf.greater_equal)
lt = migrate_opr('lt', _tf.less)
le = migrate_opr('le', _tf.less_equal)

add_n = migrate_opr('add_n', _tf.add_n)

reduce_sum = migrate_opr('reduce_sum', _tf.reduce_sum)
reduce_prod = migrate_opr('reduce_prod', _tf.reduce_prod)
reduce_mean = migrate_opr('reduce_mean', _tf.reduce_mean)
reduce_max = migrate_opr('reduce_max', _tf.reduce_max)
reduce_min = migrate_opr('reduce_min', _tf.reduce_min)
reduce_all = migrate_opr('reduce_all', _tf.reduce_all)
reduce_any = migrate_opr('reduce_any', _tf.reduce_any)

slice = migrate_opr('slice', _tf.slice)
strided_slice = migrate_opr('strided_slice', _tf.strided_slice)
split = migrate_opr('split', _tf.split)
pad = migrate_opr('pad', _tf.pad)
reverse = migrate_opr('reverse', _tf.reverse)

# array ops: creation
cast = migrate_opr('cast', _tf.cast)
zeros = migrate_opr('zeros', _tf.zeros)
ones = migrate_opr('ones', _tf.ones)
zeros_like = migrate_opr('zeros_like', _tf.zeros_like)
ones_like = migrate_opr('ones_like', _tf.ones_like)
one_hot = migrate_opr('one_hot', _tf.one_hot)
range = migrate_opr('range', _tf.range)

# array ops: advanced
where = migrate_opr('where', _tf.where)
meshgrid = migrate_opr('meshgrid', _tf.meshgrid)

# control
zero_grad = migrate_opr('zero_grad', _tf.stop_gradient)
group = migrate_opr('group', _tf.group)

# softmax related
softmax = migrate_opr('softmax', _tf.nn.softmax)
softmax_cross_entropy_with_logits = migrate_opr('softmax_cross_entropy_with_logits', _tf.nn.softmax_cross_entropy_with_logits)
sparse_softmax_cross_entropy_with_logits = migrate_opr('sparse_softmax_cross_entropy_with_logits', _tf.nn.sparse_softmax_cross_entropy_with_logits)
sigmoid_cross_entropy_with_logits = migrate_opr('sigmoid_cross_entropy_with_logits', _tf.nn.sigmoid_cross_entropy_with_logits)

# clipping
clip_by_value = migrate_opr('clip_by_value', _tf.clip_by_value)
clip_by_norm = migrate_opr('clip_by_norm', _tf.clip_by_norm)
clip_by_global_norm = migrate_opr('clip_by_global_norm', _tf.clip_by_global_norm)
clip_by_average_norm = migrate_opr('clip_by_average_norm', _tf.clip_by_average_norm)

# assignment
assign = migrate_opr('assign', _tf.assign)
