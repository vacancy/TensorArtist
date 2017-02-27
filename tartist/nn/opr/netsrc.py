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
from .helper import device_context, wrap_varnode_func
from ..graph.env import get_default_env, get_default_net
from ..graph.node import as_tftensor, as_varnode, OprNode
from ..tfutils import assign_variable, fetch_variable
from ...core.utils.meta import assert_notnone

__all__ = ['placeholder', 'variable', 'constant']


@wrap_varnode_func
def placeholder(name, shape=None, dtype=__default_dtype__, device=None):
    with device_context(device):
        var = as_varnode(tf.placeholder(name=name, shape=shape, dtype=dtype))
        get_default_net().add_to_collection(var, 'placeholders')
        return var


class VariableOp(OprNode):
    def __init__(self, name, output, owner_net):
        super().__init__(name, [], [output])
        self._owner_net = owner_net

    @property
    def owner_env(self):
        return self._owner_net.owner_env

    def get_value(self):
        var = self.outputs[0].impl
        return fetch_variable(var, self.owner_env.session)

    def set_value(self, value, use_locking=False):
        var = self.outputs[0].impl
        assign_variable(var, value, self.owner_env.session, use_locking=use_locking)
        return self


@wrap_varnode_func
def variable(name, value_or_initializer, shape=None, dtype=__default_dtype__, device=None, trainable=True):
    with device_context(device):
        if isinstance(value_or_initializer, np.ndarray):
            var = tf.Variable(initial_value=value_or_initializer, trainable=trainable, name=name, dtype=dtype)
        else:
            assert_notnone(shape, name='shape')
            var = tf.get_variable(name, shape=shape, dtype=dtype, initializer=value_or_initializer,
                                  trainable=trainable)
        var = as_varnode(var)
        get_default_net().add_to_collection(var, 'variables')
        get_default_net().add_to_collection(var.impl.initializer, 'variables/initializer')
        return VariableOp(name, var, get_default_net()).outputs[0]


@wrap_varnode_func
def constant(value, shape=None, dtype=None, name='const', device=None, verify_shape=False):
    with device_context(device):
        return tf.constant(value, dtype=dtype, shape=shape, name=name, verify_shape=verify_shape)

