# -*- coding:utf8 -*-
# File   : helper.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/31/16
# 
# This file is part of TensorArtist.

from .. import TArtGraphKeys
from ..graph.env import get_default_env, reuse_context
from ..graph.node import OprNode, as_varnode, as_tftensor
from ...core.logger import get_logger
from ...core.utils.context import EmptyContext
from ...core.utils.shape import get_2dshape, get_4dshape
import tensorflow as tf
import functools
import inspect
import contextlib

logger = get_logger(__file__)

__all__ = [
    'device_context', 
    'as_varnode', 'as_tftensor',
    'get_2dshape', 'get_4dshape', 
    'wrap_varnode_func', 'wrap_simple_named_op', 'wrap_named_op', 'wrap_named_class_func',
    'unique_opr_name', 'StaticDynamicDim', 'lazy_O', 'auto_reuse'
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


def wrap_simple_named_op(*args, use_scope=True, default_name=None):
    # do instant-binding
    def wrapper(func, default_name=default_name):
        if default_name is None:
            sig = inspect.signature(func)
            default_name = sig.parameters['name'].default

        @functools.wraps(func)
        def new_func(*args, name=default_name, **kwargs):
            if name is None:
                name = default_name
            if use_scope:
                with tf.name_scope(name):
                    return func(*args, name=name, **kwargs)
            else:
                return func(*args, name=name, **kwargs)
        return new_func
    if len(args) == 1 and callable(args[0]):
        return wrapper(args[0])
    return wrapper


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


def wrap_named_class_func(*args, in_class=True):
    def wrapper(func):
        @functools.wraps(func)
        def new_func(self, *args, **kwargs):
            opr_name = unique_opr_name(self.name)
            if opr_name not in get_default_env().get_name_scope():
                with tf.variable_scope(opr_name + '/' + func.__name__):
                    return func(self, *args, **kwargs)
            else:
                return func(self, *args, **kwargs)
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


def _argscope_wrapped(f, **default_kwargs):
    # TODO(MJY): default_kwargs must be provided as kwargs if you want to overwrite
    signature = inspect.signature(f)
    all_param_names = list(signature.parameters)

    @functools.wraps(f)
    def new_func(*args, **kwargs):
        used_default_kwargs = set()
        for i in range(len(args)):
            used_default_kwargs.add(all_param_names[i])
        for k in default_kwargs:
            if k not in kwargs:
                if k not in used_default_kwargs:
                    kwargs[k] = default_kwargs[k]
                else:
                    logger.warn('Overwrite argscope binded kwargs using args: func_name={}, param_name={}'.format(
                        f.__name__, k
                    ))
        return f(*args, **kwargs)
    return new_func


@contextlib.contextmanager
def argscope(*funcs, **kwargs):
    from tartist.nn import opr as O

    for f in funcs:
        fname = f.__name__
        assert hasattr(O, fname)
        new_f = _argscope_wrapped(f, **kwargs)
        setattr(O, fname, new_f)
    yield
    for f in funcs:
        fname = f.__name__
        setattr(O, fname, f)


class AllOprGetter(object):
    def __getattr__(self, item):
        from .. import opr as O
        return getattr(O, item)

lazy_O = AllOprGetter()


def auto_reuse(func):
    used_graphs = set()

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        g = tf.get_default_graph()
        if g not in used_graphs:
            used_graphs.add(g)
            reuse = False
        else:
            reuse = True

        with reuse_context(activate=reuse):
            return func(*args, **kwargs)

    return wrapped
