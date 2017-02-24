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
from ..graph.env import get_default_net
from ..graph.node import as_varnode
from ...core.utils.meta import assert_notnone


@wrap_varnode_func
def placeholder(name, shape=None, dtype=__default_dtype__, device=None):
    with device_context(device):
        var = as_varnode(tf.placeholder(name=name, shape=shape, dtype=dtype))
        get_default_net().add_to_collection(var, 'placeholders')
        return var


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
        return var
