# -*- coding:utf8 -*-
# File   : image_proc.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/27/17
# 
# This file is part of TensorArtist

from .helper import wrap_varnode_func
from .tensor import stack
from ..graph.node import as_varnode, __valid_tensor_types__
import tensorflow as tf
import functools
import collections

__all__ = [
    'crop_center', 'crop_lu',
    'pad_center', 'pad_rb', 'pad_rb_multiple_of',
    'img_flip', 'img_inverse'
]


def format_vi(x):
    return as_varnode(x)


def get_vi_2dshape(shape):
    """get varnode/int 2dshape"""

    if isinstance(shape, __valid_tensor_types__):
        return as_varnode(shape)
    if isinstance(shape, collections.Iterable):
        h, w = map(functools.partial(as_varnode, dtype=tf.int32), shape)
        return h, w
    v = as_varnode(int(shape), dtype=tf.int32)
    return v, v


@wrap_varnode_func
def _crop(inpvar, shape, method='center'):
    assert method in ('center', 'leftup')
    assert inpvar.partial_shape is None or len(inpvar.static_shape) == 4

    inpvar = as_varnode(shape)
    shape = get_vi_2dshape(shape)
    h, w = shape[0], shape[1]
    y = 0 if method == 'leftup' else (inpvar.shape[2] - h) // 2
    x = 0 if method == 'leftup' else (inpvar.shape[3] - w) // 2
    return inpvar[:, :, y:y+h, x:x+w]


crop_center = functools.partial(_crop, method='center')
crop_lu = functools.partial(_crop, method='leftup')


@wrap_varnode_func
def _pad(inpvar, shape, method='center', mode='CONSTANT'):
    assert method in ('center', 'rightbottom')
    assert inpvar.static_shape is not None and len(inpvar.static_shape) == 4

    shape = get_vi_2dshape(shape)
    h, w = inpvar.shape[2], inpvar.shape[3]
    y0 = 0 if method == 'rightbottom' else (shape[0] - h) // 2
    x0 = 0 if method == 'rightbottom' else (shape[1] - w) // 2
    y1 = shape[0] - h - y0
    x1 = shape[1] - w - x0

    new_shape = [inpvar.static_shape[0], None, None, inpvar.static_shape[3]]
    out = tf.pad(inpvar, stack([[0, 0], stack([x0, x1]), stack([y0, y1]), [0, 0]]), mode=mode)
    out.set_shape(new_shape)
    return out


pad_center = functools.partial(_pad, method='center')
pad_rb = functools.partial(_pad, method='rightbottom')


@wrap_varnode_func
def pad_rb_multiple_of(inpvar, multiple, val=0):
    assert inpvar.static_shape is None or len(inpvar.static_shape) == 4

    multiple = get_vi_2dshape(multiple)
    h, w = inpvar.shape[2], inpvar.shape[3]

    def canonicalize(x, mul):
        a = x // mul
        return (a + 1 - tf.cast(tf.equal(a * mul, x), tf.int32)) * mul
    return pad_rb(inpvar, [canonicalize(h, multiple[0]), canonicalize(w, multiple[1])])


# def resize(name, inpvar, size=None, scale=None):
#     assert size is not None or scale is not None
#     assert inpvar.partial_shape is None or inpvar.partial_shape.ndim == 4
#
#     w_opr = mgsk_opr.ConstProvider(numpy.eye(3, dtype='float32').reshape((1, 3, 3)),
#             name='{}_W'.format(name),  comp_node=inpvar.comp_node)
#     w = w_opr.outputs[0]
#
#     if size is not None:
#         w = w.set_sub[0, 0, 0](inpvar.shape[3] / size[1])
#         w = w.set_sub[0, 1, 1](inpvar.shape[2] / size[0])
#     else:
#         w = w.set_sub[0, 0, 0](w[0, 0, 0] / scale[1])
#         w = w.set_sub[0, 1, 1](w[0, 1, 1] / scale[0])
#         size = inpvar.shape[2:4] * scale
#
#     opr = mgsk_opr.WarpPerspective(name, inpvar, w, size)
#     return opr.outputs[0]


@wrap_varnode_func
def img_inverse(inpvar):
    return 255 -inpvar


@wrap_varnode_func
def img_flip(inpvar, axis=2):
    assert inpvar.static_shape is None or len(inpvar.static_shape) == 4
    assert axis in (1, 2)
    if axis == 1:
        return inpvar[:, ::-1, :, :]
    return inpvar[:, :, ::-1, :]
