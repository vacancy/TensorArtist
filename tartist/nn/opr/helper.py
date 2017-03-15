# -*- coding:utf8 -*-
# File   : helper.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/31/16
# 
# This file is part of TensorArtist

from .. import TArtGraphKeys
from ..graph.node import OprNode, as_varnode
from ...core.utils.context import EmptyContext
from ...core.utils.shape import get_2dshape, get_4dshape
import tensorflow as tf
import functools

__all__ = [
    'device_context', 
    'get_2dshape', 'get_4dshape', 
    'wrap_varnode_func', 'wrap_named_op', 'unique_opr_name',
    'StaticDynamicDim'
]


def device_context(device=None):
    if device is None:
        return EmptyContext()
    else:
        return tf.device(device)


def wrap_varnode_func(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        outputs = func(*args, **kwargs)
        if isinstance(outputs, (tuple, list)):
            return tuple(map(as_varnode, outputs))
        return as_varnode(outputs)
    return new_func


def wrap_named_op(*args, use_scope=True):
    def wrapper(func):
        @functools.wraps(func)
        def new_func(name, *args, **kwargs):
            opr_name = unique_opr_name(name) 
            if use_scope:
                with tf.variable_scope(name):
                    outputs = func(name, *args, **kwargs)
            else:
                outputs = func(name, *args, **kwargs)
            
            opr = OprNode(opr_name)
            if isinstance(outputs, (tuple, list)):
                outputs = tuple(map(as_varnode, outputs))
                for o in outputs:
                    if o.taop is None:
                        o.set_taop(opr)
                tf.add_to_collection(TArtGraphKeys.TART_OPERATORS, outputs[0].taop)
            else:
                outputs = as_varnode(outputs)
                if outputs.taop is None:
                    outputs.set_taop(opr)
                tf.add_to_collection(TArtGraphKeys.TART_OPERATORS, outputs.taop)
            return outputs
        return new_func
    if len(args) == 1 and callable(args[0]):
        return wrapper(args[0])
    return wrapper


def unique_opr_name(name):
    return tf.get_default_graph().unique_name(name, mark_as_used=False)


class StaticDynamicDim(object):
    """Enable shape computation for both static and dynamic shape with unknown value (None)
    https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/models/conv2d.py
    """

    def __init__(self, static, dynamic):
        self.static = static
        self.dynamic = dynamic

    def op(self, func):
        try:
            new_static = func(self.static)
            return StaticDynamicDim(new_static, new_static)
        except:
            return StaticDynamicDim(None, func(self.static))

    def __add__(self, other):
        return self.op(lambda v: v + other)

    def __radd__(self, other):
        return self.op(lambda v: other + v)

    def __mul__(self, other):
        return self.op(lambda v: v * other)

    def __rmul__(self, other):
        return self.op(lambda v: other * v)
