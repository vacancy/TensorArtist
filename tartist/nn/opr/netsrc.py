# -*- coding:utf8 -*-
# File   : netsrc.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/31/16
# 
# This file is part of TensorArtist

import numpy as np
import tensorflow as tf

from ._defaults import __default_dtype__
from .helper import device_context, wrap_varnode_func, wrap_named_op, unique_opr_name
from ..graph.env import get_default_env
from ..graph.node import as_varnode, OprNode, __valid_tensor_types__
from ..tfutils import assign_variable, fetch_variable
from ...core.utils.meta import assert_notnone

__all__ = ['placeholder', 'variable', 'scalar', 'constant', 'ensure_variable']


@wrap_varnode_func
def placeholder(name, shape=None, dtype=__default_dtype__, device=None):
    with device_context(device):
        var = as_varnode(tf.placeholder(name=name, shape=shape, dtype=dtype))
        return var


class VariableOp(OprNode):
    def __init__(self, name, output, owner_env):
        super().__init__(name, [], [output])
        self._owner_env = owner_env

    def get_value(self):
        var = self.outputs[0].impl
        return fetch_variable(var, self._owner_env.session)

    def set_value(self, value, use_locking=False):
        var = self.outputs[0].impl
        assign_variable(var, value, self._owner_env.session, use_locking=use_locking)
        return self


@wrap_named_op(use_scope=False)
@wrap_varnode_func
def variable(name, value_or_initializer, shape=None, dtype=__default_dtype__, device=None, trainable=True,
             collections=None):
    opr_name = unique_opr_name(name) 

    with device_context(device):
        if isinstance(value_or_initializer, (np.ndarray, float, int)):
            if type(value_or_initializer) is float:
                dtype = dtype or tf.float32
            elif type(value_or_initializer) is int:
                dtype = dtype or tf.int32
            var = tf.Variable(initial_value=value_or_initializer, trainable=trainable, name=name, dtype=dtype,
                              collections=collections)
        else:
            assert_notnone(shape, name='shape')
            var = tf.get_variable(name, shape=shape, dtype=dtype, initializer=value_or_initializer,
                                  trainable=trainable, collections=collections)
        var = as_varnode(var)
        return VariableOp(opr_name, var, get_default_env()).outputs[0]


def scalar(name, value, dtype=__default_dtype__, device=None, trainable=False, collections=None):
    return variable(name, value, shape=0, dtype=dtype, device=device, trainable=trainable, collections=collections)


@wrap_varnode_func
def constant(value, shape=None, dtype=None, name='const', device=None, verify_shape=False):
    with device_context(device):
        return tf.constant(value, dtype=dtype, shape=shape, name=name, verify_shape=verify_shape)


def ensure_variable(name, value_or_intializer, *args, **kwargs):
    if not isinstance(value_or_intializer, __valid_tensor_types__):
        return variable(name, value_or_intializer, *args, **kwargs)
    return value_or_intializer
