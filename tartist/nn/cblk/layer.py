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

__all__ = ['conv2d_kaiming', 'bn_relu_conv', 'conv_bn_relu', 'global_avg_pooling2d']


def conv2d_kaiming(name, src, channel, kernel, stride=1, padding='SAME', use_bias=True, nonlin=O.identity):
    kernel = get_2dshape(kernel)
    fan_out = kernel[0] * kernel[1] * channel

    W_init = O.truncated_normal_initializer(stddev=sqrt(2./fan_out))
    out = O.conv2d(name, src, channel, kernel, stride=stride, padding=padding, 
            use_bias=use_bias, nonlin=nonlin, W=W_init)

    return out


def bn_relu_conv(name, src, channel, kernel, stride=1, padding='SAME', use_bias=False):
    _ = src
    _ = O.bn_relu(_, name=name + '_bn_relu')
    _ = conv2d_kaiming(name, _, channel, kernel, stride, padding, use_bias=use_bias) 
    return _


def conv_bn_relu(name, src, channel, kernel, stride=1, padding='SAME', use_bias=False):
    _ = src
    _ = conv2d_kaiming(name, _, channel, kernel, stride, padding, use_bias=use_bias)
    _ = O.bn_relu(_, name=name + '_bn_relu')
    return _


def global_avg_pooling2d(name, src):
    assert src.ndims == 4
    return src.mean(axis=[1, 2], name=name)

