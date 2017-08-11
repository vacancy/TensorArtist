# -*- coding:utf8 -*-
# File   : netsrc.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/31/16
# 
# This file is part of TensorArtist.

import numpy as np
import tensorflow as tf

from ._defaults import __default_dtype__
from .helper import device_context, wrap_varnode_func
from ..graph.env import get_default_env
from ..graph.node import as_varnode, VarNode, __valid_tensor_types__
from ..tfutils import assign_variable, fetch_variable, TArtGraphKeys, extend_collection_list
from ...core.utils.meta import assert_notnone

__all__ = ['placeholder', 'variable', 'scalar', 'constant', 'ensure_variable', 'get_variable',
           'get_scalar', 'get_scalar_value', 'set_scalar_value']


@wrap_varnode_func
def placeholder(name, shape=None, dtype=__default_dtype__, device=None):
    with device_context(device):
        var = as_varnode(tf.placeholder(name=name, shape=shape, dtype=dtype))
        tf.add_to_collection(TArtGraphKeys.PLACEHOLDERS, var)
        return var


class MutableVarNode(VarNode):
    def __init__(self, impl, owner_env):
        super().__init__(impl)
        self._owner_env = owner_env

    def get_value(self):
        return fetch_variable(self.tft, self._owner_env.session)

    def set_value(self, value, use_locking=False):
        assign_variable(self.tft, value, self._owner_env.session, use_locking=use_locking)
        return self


@wrap_varnode_func
def variable(name, value_or_initializer, shape=None, dtype=__default_dtype__, device=None, trainable=True,
             collections=None):

    collections = extend_collection_list(collections, TArtGraphKeys.TART_VARIABLES)

    if tf.get_variable_scope().reuse:
        var = tf.get_variable(name)
        name = var.name
        tavar = get_default_env().find_in_collection_by_name(TArtGraphKeys.TART_VARIABLES, name)
        if tavar is not None:
            return tavar
    else:
        with device_context(device):
            if isinstance(value_or_initializer, (np.ndarray, float, int)):
                if type(value_or_initializer) is float:
                    dtype = dtype or tf.float32
                elif type(value_or_initializer) is int:
                    dtype = dtype or tf.int32
    
                var = tf.get_variable(name, shape=shape, dtype=dtype, initializer=value_or_initializer,
                                      trainable=trainable, collections=None)
            else:
                assert_notnone(shape, name='shape')
                var = tf.get_variable(name, shape=shape, dtype=dtype, initializer=value_or_initializer,
                                      trainable=trainable, collections=None)

    var = MutableVarNode(var, get_default_env())
    tf.get_default_graph().add_to_collections(collections, var)
    return var


@wrap_varnode_func
def scalar(name, value, dtype=__default_dtype__, device=None, trainable=False,
           collections=None, summary=False):

    collections = extend_collection_list(collections, TArtGraphKeys.SCALAR_VARIABLES)
    value = float(value)
    var = variable(name, value, shape=None, dtype=dtype, device=device, trainable=trainable, collections=collections)
    if summary:
        from .. import summary
        summary.scalar(name, var)
    return var


@wrap_varnode_func
def get_variable(name, env=None, collection=TArtGraphKeys.TART_VARIABLES):
    env = env or get_default_env()
    sym = env.find_in_collection_by_name(collection, name)
    return sym


@wrap_varnode_func
def get_scalar(name, env=None, collection=TArtGraphKeys.SCALAR_VARIABLES):
    return get_variable(name, env=env, collection=collection)


@wrap_varnode_func
def get_scalar_value(name, env=None, collection=TArtGraphKeys.SCALAR_VARIABLES):
    return get_scalar(name, env=env, collection=collection).get_value()


@wrap_varnode_func
def set_scalar_value(name, value, env=None, collection=TArtGraphKeys.SCALAR_VARIABLES):
    return get_scalar(name, env=env, collection=collection).set_value(value)


@wrap_varnode_func
def constant(value, shape=None, dtype=None, name='const', device=None, verify_shape=False):
    with device_context(device):
        return tf.constant(value, dtype=dtype, shape=shape, name=name, verify_shape=verify_shape)


@wrap_varnode_func
def ensure_variable(name, value_or_intializer, *args, **kwargs):
    if not isinstance(value_or_intializer, __valid_tensor_types__):
        return variable(name, value_or_intializer, *args, **kwargs)
    return value_or_intializer
