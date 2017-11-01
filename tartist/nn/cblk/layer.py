# -*- coding:utf8 -*-
# File   : layer.py
# Author : Jiayuan Mao
# Email  : mjy@megvii.com
# Date   : 10/29/17
#
# This file is part of TensorArtist.

from .. import opr as O
from ...core.utils.shape import get_2dshape
from numpy import sqrt
import functools

__all__ = ['conv2d_comp', 'pooling2d_comp', 'conv2d_kaiming', 'bn_relu_conv', 'conv_bn_relu', 'global_avg_pooling2d']


def _pad_same(name, src, kernel, padding):
    kernel = get_2dshape(kernel)

    if padding == 'SAME' and (kernel[0] != 1 or kernel[1] != 1):
        assert kernel[0] % 2 == 1 and kernel[1] % 2 == 1
        pad = (kernel[0] // 2, kernel[1] // 2)
        src = O.pad(src, [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]], name=name + '_pad')

    return src


def conv2d_comp(name, src, channel, kernel, stride=1, padding='SAME', use_bias=True, nonlin=O.identity, **kwargs):
    src = _pad_same(name, src, kernel, padding)
    return O.conv2d(name, src, channel, kernel, stride=stride, padding='VALID', use_bias=use_bias, nonlin=nonlin,
            **kwargs)


def pooling2d_comp(name, src, kernel, stride=None, padding='SAME', method='MAX'):
    src = _pad_same(name, src, kernel, padding)
    return O.pooling2d(name, src, kernel, stride=stride, padding='VALID', method=method)


max_pooling2d_comp = functools.partial(pooling2d_comp, method='MAX')
avg_pooling2d_comp = functools.partial(pooling2d_comp, method='AVG')


def conv2d_kaiming(name, src, channel, kernel, stride=1, padding='SAME', use_bias=True, nonlin=O.identity,
        algo=conv2d_comp):

    kernel = get_2dshape(kernel)
    fan_out = kernel[0] * kernel[1] * channel

    W_init = O.truncated_normal_initializer(stddev=sqrt(2./fan_out))
    out = algo(name, src, channel, kernel, stride=stride, padding=padding,
            use_bias=use_bias, nonlin=nonlin, W=W_init)

    return out


def bn_relu_conv(name, src, channel, kernel, stride=1, padding='SAME', use_bias=False):
    _ = src
    _ = O.batch_norm(name + '_bn', _)
    _ = O.relu(_, name=name + '_relu')
    _ = conv2d_kaiming(name, _, channel, kernel, stride, padding, use_bias=use_bias) 
    return _


def conv_bn_relu(name, src, channel, kernel, stride=1, padding='SAME', use_bias=False):
    _ = src
    _ = conv2d_kaiming(name, _, channel, kernel, stride, padding, use_bias=use_bias)
    _ = O.batch_norm(name + '_bn', _)
    _ = O.relu(_, name=name + '_relu')
    return _


def global_avg_pooling2d(name, src):
    assert src.ndims == 4
    return src.mean(axis=[1, 2], name=name)

