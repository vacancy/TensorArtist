# -*- coding:utf8 -*-
# File   : grad.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 4/13/17
#
# This file is part of TensorArtist


from .helper import lazy_O as O
from .helper import as_varnode, as_tftensor, wrap_varnode_func, wrap_simple_named_op
import tensorflow as tf
import functools

__all__ = ['clip_gradient', 'preserve_gradient_simple', 'binarize_01']


@wrap_varnode_func
def clip_gradient(inpvar, clip_value_min, clip_value_max, name='clip_gradient'):
    def _clip_gradient_backward(unused_op, grad):
        return tf.clip_by_value(grad, clip_value_min, clip_value_max)

    @function.Defun(inpvar.dtype, python_grad_func=_clip_gradient_backward, func_name="ClipGradient")
    def _clip_gradient_forward(x):
        return x

    with tf.name_scope(name, values=[inpvar]):
        out = _clip_gradient_forward(inpvar)
        as_tftensor(out).set_shape(as_tftensor(inpvar).get_shape())
    return output


def preserve_gradient_simple(func):
    @wrap_varnode_func
    @functools.wraps(func)
    def new_func(inpvar, *args, **kwargs):
        def _backward(op, grad):
            return grad

        @function.Defun(inpvar.dtype, python_grad_func=_backward, func_name=func.__name__)
        def _forward(x):
            return func(inpvar, *args, **kwargs)

        with tf.name_scope(func.__name__, values=[inpvar]):
            out = _forward(inpvar)
            as_tftensor(out).set_shape(as_tftensor(inpvar).get_shape())
        return out
    
    return new_func

