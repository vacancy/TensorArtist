# -*- coding:utf8 -*-
# File   : node.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/31/16
# 
# This file is part of TensorArtist

from ...core.utils.meta import assert_instance, assert_notnone
import tensorflow as tf

__all__ = ['VarNode', 'as_varnode', 'as_tftensor']

__valid_tf_tensor_types__ = (tf.Tensor, tf.Variable)


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


class VarNode(object):
    def __init__(self, impl):
        self.__impl = impl
        self.__taop = None
        assert_instance(impl, __valid_tf_tensor_types__)

    @property
    def impl(self):
        return self.__impl

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
    def __init__(self, name):
        self.__name = tf.get_default_graph().unique_name(name, mark_as_used=False)

    @property
    def name(self):
        return self.__name


def as_varnode(tensor):
    if isinstance(tensor, VarNode):
        return tensor
    return varnode_store.get(tensor)


def as_tftensor(tensor):
    if isinstance(tensor, __valid_tf_tensor_types__):
        return tensor
    assert_instance(tensor, VarNode)
    return tensor.impl

