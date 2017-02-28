# -*- coding:utf8 -*-
# File   : node.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/31/16
# 
# This file is part of TensorArtist

from ...core.utils.meta import assert_instance, assert_notnone, AttrObject

import numpy as np
import tensorflow as tf

__all__ = [
    '__valid_tensor_types__', '__valid_tf_tensor_types__',
    'VarNode', 'as_varnode', 'as_tftensor'
]


class VarNodeStore(object):
    def __init__(self):
        self.__kv = {}

    def get(self, tensor):
        if tensor in self.__kv:
            return self.__kv[tensor]
        v = self._make(tensor)
        self.__kv[tensor] = v
        return v

    def _make(self, tensor):
        return VarNode(tensor)

varnode_store = VarNodeStore()


class VarNodeOpDecl(object):
    def __binary(self, rhs, op_name):
        rhs = as_varnode(rhs)
        return as_varnode(getattr(tf, op_name)(self, rhs))

    def __rbinary(self, lhs, op_name):
        lhs = as_varnode(lhs)
        return as_varnode(getattr(tf, op_name)(lhs, self))

    def __add__(self, rhs):
        return self.__binary(rhs, 'add')
    def __radd__(self, lhs):
        return self.__rbinary(lhs, 'add')

    def __sub__(self, rhs):
        return self.__binary(rhs, 'subtract')
    def __rsub__(self, lhs):
        return self.__rbinary(lhs, 'subtract')

    def __mul__(self, rhs):
        return self.__binary(rhs, 'multiply')
    def __rmul__(self, lhs):
        return self.__rbinary(lhs, 'multiply')

    def __matmul__(self, rhs):
        return self.__binary(rhs, 'matmul')
    def __rmatmul__(self, lhs):
        return self.__rbinary(lhs, 'matmul')

    def __truediv__(self, rhs):
        return self.__binary(rhs, 'truediv')
    def __rtruediv__(self, lhs):
        return self.__rbinary(lhs, 'truediv')

    def __floordiv__(self, rhs):
        return self.__binary(rhs, 'floordiv')
    def __rfloordiv__(self, lhs):
        return self.__rbinary(lhs, 'floordiv')

    def __mod__(self, rhs):
        return self.__binary(rhs, 'mod')
    def __rmod__(self, lhs):
        return self.__rbinary(lhs, 'mod')

    def __pow__(self, rhs):
        return self.__binary(rhs, 'pow')
    def __rpow__(self, lhs):
        return self.__rbinary(lhs, 'pow')

    def __lt__(self, rhs):
        return self.__binary(rhs, 'less')

    def __le__(self, rhs):
        return self.__binary(rhs, 'less_equal')

    def __gt__(self, rhs):
        return self.__binary(rhs, 'greater')

    def __ge__(self, rhs):
        return self.__binary(rhs, 'greater_equal')

    # do not define __eq__ because it would make the node unhashable
    def eq(self, rhs):
        return self.__binary(rhs, 'equal')

    def neq(self, rhs):
        return self.__binary(rhs, 'not_equal')

    def __neg__(self):
        return as_varnode(tf.neg(self))

    def __getitem__(self, slices):
        from ..opr.tensor import NormalSlice
        return NormalSlice(self)[slices]
  
    @property
    def sub(self):
        from ..opr.tensor import NormalSlice
        return NormalSlice(self)

    @property
    def set_sub(self):
        from ..opr.tensor import NormalSliceSetter
        return NormalSliceSetter(self)

    @property
    def ai(self):
        from ..opr.tensor import AdvancedSlice
        return AdvancedSlice(self)

    @property
    def set_ai(self):
        from ..opr.tensor import AdvancedSliceSetter
        return AdvancedSliceSetter(self)

    def __iter__(self):
        raise ValueError('iterating over {} is not allowed'.format(type(self).__name__))

    def astype(self, dtype):
        return as_varnode(tf.cast(self, dtype))

    def reshape(self, *tshape, name=None):
        if len(tshape) == 1:
            tshape, = tshape
        from ..opr.shape import reshape
        return as_varnode(reshape(self, tshape=tshape, name=name))

    def broadcast(self, *tshape, name=None):
        if len(tshape) == 1:
            tshape, = tshape
        from ..opr.shape import broadcast
        return as_varnode(broadcast(self, tshape=tshape, name=name))

    def dimshuffle(self, *pattern, name=None):
        from ..opr.shape import dimshuffle 
        return as_varnode(dimshuffle(self, perm=pattern, name=name))

    def add_axis(self, axis):
        from ..opr.shape import add_axis 
        return as_varnode(add_axis(self, axis=axis))

    def remove_axis(self, axis):
        from ..opr.shape import remove_axis 
        return as_varnode(remove_axis(axis=axis))

    def sum(self, axis=None, keepdims=False, name=None):
        return as_varnode(tf.reduce_sum(self, axis=axis, keep_dims=keepdims, name=name))

    def mean(self, axis=None, keepdims=False, name=None):
        return as_varnode(tf.reduce_mean(self, axis=axis, keep_dims=keepdims, name=name))

    def max(self, axis=None, keepdims=False, name=None):
        return as_varnode(tf.reduce_max(self, axis=axis, keep_dims=keepdims, name=name))

    def min(self, axis=None, keepdims=False, name=None):
        return as_varnode(tf.reduce_min(self, axis=axis, keep_dims=keepdims, name=name))

    def argmax(self, axis=None, name=None):
        return as_varnode(tf.argmax(self, axis=axis, name=name))

    def argmin(self, axis=None, name=None):
        return as_varnode(tf.argmin(self, axis=axis, name=name))

    def prod(self, axis=None, keepdims=False, name=None):
        return as_varnode(tf.reduce_prod(self, axis=axis, keep_dims=keepdims, name=name))

    def std(self):
        from ..opr.arith import std
        return std(self)

    def rms(self):
        from ..opr.arith import rms
        return rms(self)

    def flatten(self):
        from ..opr.shape import flatten
        return flatten(self)

    def flatten2(self):
        from ..opr.shape import flatten2
        return flatten2(self)

    def eval(self, session=None, feed_dict=None, **kwargs):
        from .env import get_default_env
        from .function import Function 

        session = session or get_default_env().session
        feed_dict = feed_dict or {}
        feed_dict.update(kwargs)
        feed_dict = Function.canonize_feed_dict(feed_dict)
        return as_tftensor(self).eval(feed_dict=feed_dict, session=session)


class VarNode(VarNodeOpDecl):
    class Flags(AttrObject):
        data_parallel_reduce_method = 'CONCAT'

    def __init__(self, impl, **flags):
        self.__impl = impl
        self.__taop = None
        self.__flags = type(self).Flags(**flags)
        assert_instance(impl, __valid_tf_tensor_types__)

    @property
    def impl(self):
        return self.__impl

    @property
    def flags(self):
        return self.__flags

    @property
    def static_shape(self):
        ss = self.__impl.get_shape()
        if ss.ndims is None:
            return None
        return tuple(ss.as_list())

    @property
    def ndims(self):
        return self.__impl.get_shape().ndims

    @property
    def shape(self):
        return tf.shape(self.__impl)

    @property
    def dtype(self):
        return self.__impl.dtype

    @property
    def name(self):
        return self.__impl.name

    @property
    def op(self):
        return self.__impl.op

    @property
    def taop(self):
        return self.__taop

    def set_taop(self, op):
        self.__taop = op
        assert_instance(op, OprNode)


class OprNode(object):
    def __init__(self, name, inputs=[], outputs=[]):
        self.__name = tf.get_default_graph().unique_name(name, mark_as_used=False)
        self.__inputs = inputs
        self.__outputs = outputs

        self.__mark_outputs()

    @property
    def name(self):
        return self.__name

    @property
    def inputs(self):
        return self.__inputs

    def set_inputs(self, inputs):
        self.__inputs = inputs
        return self

    @property
    def outputs(self):
        return self.__outputs

    def add_output(self, output):
        self.__outputs.append(output)
        self.__mark_outputs()
        return self

    def __mark_outputs(self):
        for i, o in enumerate(self.__outputs):
            self.__outputs[i] = as_varnode(o)
            self.__outputs[i].set_taop(self)


__valid_tensor_types__ = (VarNode, tf.Tensor, tf.Variable, tf.Operation)
__valid_tf_tensor_types__ = (tf.Tensor, tf.Variable, tf.Operation)


def infer_dtype_from_const(v):
    def canonize(dtype):
        if dtype == np.float64:
            dtype = np.float32
    if isinstance(v, np.ndarray):
        dtype = v.dtype
    if isinstance(v, int) or (isinstance(v, (tuple, list)) and isinstance(v[0], int)):
        return np.int32
    return np.float32

def as_varnode(tensor, dtype=None):
    if isinstance(tensor, VarNode):
        return tensor

    if isinstance(tensor, (np.ndarray, int, float, tuple, list)):
        dtype = dtype or infer_dtype_from_const(tensor) 
        from ..opr.netsrc import constant
        return constant(tensor, dtype=dtype)

    return varnode_store.get(tensor)


def as_tftensor(tensor):
    if isinstance(tensor, __valid_tf_tensor_types__):
        return tensor
    assert_instance(tensor, VarNode)
    return tensor.impl

