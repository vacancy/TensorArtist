# -*- coding:utf8 -*-
# File   : resnet_preact.py
# Author : Jiayuan Mao
# Email  : mjy@megvii.com
# Date   : 10/29/17
#
# This file is part of TensorArtist.

from . import layer
from .. import opr as O
import functools

__all__ = ['make_resnet', 'make_resnet_18', 'make_resnet_34', 'make_resnet_50',
        'make_resnet_101', 'make_resnet_152']


def residual_first(name, src, increase_dim=False, is_bottleneck=False):
    _ = src
    shape = _.static_shape

    if not is_bottleneck:
        in_channel = shape[3]
        out_channel = in_channel
    else:
        in_channel = shape[3]
        out_channel = in_channel * 4

    if not is_bottleneck:
        _ = layer.conv_bn_relu('{}_branch2a'.format(name), _, out_channel, 3, stride=1)
        _ = layer.conv2d_kaiming('{}_branch2b'.format(name), _, out_channel, 3, stride=1, use_bias=False)
    else:
        _ = layer.conv_bn_relu('{}_branch2a'.format(name), _, in_channel, 1, stride=1)
        _ = layer.conv_bn_relu('{}_branch2b'.format(name), _, in_channel, 3, stride=1)
        _ = layer.conv2d_kaiming('{}_branch2c'.format(name), _, out_channel, 1, stride=1, use_bias=False)

        src = layer.conv2d_kaiming('{}_branch1'.format(name), src, out_channel, 1, stride=1, use_bias=False)

    _ = O.add(_, src, name='{}_addition'.format(name))

    return _


def residual_block(name, src, increase_dim=False, is_bottleneck=False):
    _ = src
    shape = _.static_shape

    if increase_dim:
        if not is_bottleneck:
            in_channel = shape[3]
            out_channel = in_channel * 2
        else:
            in_channel = shape[3] // 2
            out_channel = in_channel * 4
        stride = 2
    else:
        if not is_bottleneck:
            in_channel = shape[3]
            out_channel = in_channel
        else:
            in_channel = shape[3] // 4
            out_channel = in_channel * 4
        stride = 1

    if not is_bottleneck:
        _ = layer.bn_relu_conv('{}_branch2a'.format(name), _, out_channel, 3, stride=stride)
        _ = layer.bn_relu_conv('{}_branch2b'.format(name), _, out_channel, 3, stride=1)
    else:
        _ = layer.bn_relu_conv('{}_branch2a'.format(name), _, in_channel, 1, stride=stride)
        _ = layer.bn_relu_conv('{}_branch2b'.format(name), _, in_channel, 3, stride=1)
        _ = layer.bn_relu_conv('{}_branch2c'.format(name), _, out_channel, 1, stride=1)

    if increase_dim:
        src = layer.conv2d_kaiming('{}_branch1'.format(name), src, out_channel, 1, stride=2, use_bias=False)

    _ = O.add(_, src, name='{}_addition'.format(name))

    return _


def make_resnet(src, blocks, is_bottleneck, output_imm=False, imm_act=False):
    """
    Build resnet (preact version).

    :param src: input tensor, of data type NHWC.
    :param blocks: number of residual blocks for each residual module. Length should be 4.
    :param is_bottleneck: use the bottleneck module or not (1x1 -> 3x3 -> 1x1).
    :param output_imm: whether to output immediate results (conv1 ~ conv5) or not. If true, return value would be `gap, [conv1, conv2, conv3, conv4, conv5]`.
    :param imm_act: whether to add `bn_relu` as activation function for each immediate convs.
    """

    _ = src
    _ = layer.conv2d_kaiming('conv1', _, 64, 7, stride=2, use_bias=False)
    convs_imm = [_]
    _ = O.batch_norm('conv1_bn', _)
    _ = O.relu(_, name='conv1_relu')
    convs_imm_act = [_]
    _ = layer.max_pooling2d_comp('pool1', _, 3, stride=2, padding='SAME')

    residual = functools.partial(residual_block, is_bottleneck=is_bottleneck)

    assert len(blocks) == 4
    stages = [2, 3, 4, 5]

    for s, b in zip(stages, blocks):
        if s == 2:
            _ = residual_first('conv2_0', _, is_bottleneck=is_bottleneck)
        else:
            _ = residual('conv{}_0'.format(s), _, increase_dim=True)
        print(_.name, _.static_shape)
        for i in range(1, b):
            _ = residual('conv{}_{}'.format(s, i), _)
            print(_.name, _.static_shape)

        convs_imm.append(_)
        if imm_act:
            _act = O.bn_relu(_, name='conv{}_act'.format(s))
            convs_imm_act.append(_act)

    _ = O.batch_norm('gap_bn', _)
    _ = O.relu(_, name='gap_relu')
    _ = layer.global_avg_pooling2d('gap', _)

    if output_imm:
        if imm_act:
            return _, convs_imm_act
        else:
            return _, convs_imm

    return _


make_resnet_18 = functools.partial(make_resnet, blocks=[2, 2, 2, 2], is_bottleneck=False)
make_resnet_34 = functools.partial(make_resnet, blocks=[3, 4, 6, 3], is_bottleneck=False)
make_resnet_50 = functools.partial(make_resnet, blocks=[3, 4, 6, 3], is_bottleneck=True)
make_resnet_101 = functools.partial(make_resnet, blocks=[3, 4, 23, 3], is_bottleneck=True)
make_resnet_152 = functools.partial(make_resnet, blocks=[3, 8, 36, 3], is_bottleneck=True)


if __name__ == '__main__':
    img = O.placeholder('img', shape=(512, 224, 224, 3))
    # out = make_resnet(img, [2, 2, 2, 2], False)
    out = make_resnet(img, [3, 4, 6, 3], is_bottleneck=True)

    from tartist.nn import get_default_env
    for i in get_default_env().graph.get_operations():
        print(i.name, i.type, sep='@', end=': ')
        print(*[x.name for x in i.inputs], sep=', ')

