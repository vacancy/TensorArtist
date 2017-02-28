# -*- coding:utf8 -*-
# File   : helper.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/31/16
# 
# This file is part of TensorArtist

from ..graph.env import get_default_net
from ..graph.node import OprNode, as_varnode
from ...core.utils.context import EmptyContext
import tensorflow as tf
import functools
import collections

__all__ = [
    'device_context', 
    'get_2dshape', 'get_4dshape', 
    'wrap_varnode_func', 'wrap_named_op', 'unique_opr_name'
]


def device_context(device=None):
    if device is None:
        return EmptyContext()
    else:
        return tf.device(device)


def get_2dshape(x, default=None):
    if x is None:
        return default
    if isinstance(x, collections.Iterable):
        x = tuple(x)
        if len(x) == 1:
            return x[0], x[0]
        else:
            assert len(x) == 2, '2dshape must be of length 1 or 2'
            return x
    else:
        x = int(x)
        return x, x


def get_4dshape(x, default=None):
    if x is None:
        return default
    if isinstance(x, collections.Iterable):
        x = tuple(x)
        if len(x) == 1:
            return 1, x[0], x[0], 1
        elif len(x) == 2:
            return 1, x[0], x[1], 1
        else:
            assert len(x) == 4, '2dshape must be of length 1, 2, or 4'
            return x
    else:
        x = int(x)
        return 1, x, x, 1


def wrap_varnode_func(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        outputs = func(*args, **kwargs)
        if isinstance(outputs, (tuple, list)):
            return tuple(map(as_varnode, outputs))
        return as_varnode(outputs)
    return new_func


def wrap_named_op(func):
    @functools.wraps(func)
    def new_func(name, *args, **kwargs):
        opr_name = unique_opr_name(name) 
        with tf.variable_scope(name):
            outputs = func(name, *args, **kwargs)
        
        opr = OprNode(opr_name)
        get_default_net().add_to_collection(opr, 'oprnodes')
        if isinstance(outputs, (tuple, list)):
            outputs = tuple(map(as_varnode, outputs))
            for o in outputs:
                if o.taop is None:
                    o.set_taop(opr)
        else:
            outputs = as_varnode(outputs)
            if outputs.taop is None:
                outputs.set_taop(opr)
        return outputs
    return new_func


def unique_opr_name(name):
    return tf.get_default_graph().unique_name(name, mark_as_used=False)

