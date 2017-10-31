# -*- coding:utf8 -*-
# File   : resnet.py
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


def create_bn_relu(name, inpvar, channel, kernel, stride, padding='SAME', has_bn=True, has_relu=True):
    _ = inpvar
    _ = O.conv2d(name, _, channel, kernel, stride=stride, padding=padding)
    if has_bn:
        _ = O.batch_norm(name + '_bn', _)
    if has_relu:
        _ = O.relu(_, name=name + '_relu')
    return _


def create_bottleneck(prefix, inpvar, stride, nr_outputs1, nr_outputs2, has_proj=False):
    proj = inpvar 
    if has_proj:
        proj = create_bn_relu('conv' + prefix + '_branch1', proj, nr_outputs2, 1, stride=stride, has_relu=False)

    _ = inpvar
    _ = create_bn_relu('conv' + prefix + '_branch2a', _, nr_outputs1, 1, stride=stride, has_relu=True)
    _ = create_bn_relu('conv' + prefix + '_branch2b', _, nr_outputs1, 3, stride=1, has_relu=True)
    _ = create_bn_relu('conv' + prefix + '_branch2c', _, nr_outputs2, 1, stride=1, has_relu=False)
    _ = _ + proj
    return O.relu(_, name='relu' + prefix)


def create_block(prefix, inpvar, stride, nr_outputs1, nr_outputs2, has_proj=False):
    proj = inpvar 
    if has_proj:
        proj = create_bn_relu('conv' + prefix + '_branch1', proj, nr_outputs2, 1, stride=stride, has_relu=False)

    _ = inpvar
    _ = create_bn_relu('conv' + prefix + '_branch2a', _, nr_outputs1, 3, stride=stride, has_relu=True)
    _ = create_bn_relu('conv' + prefix + '_branch2b', _, nr_outputs1, 3, stride=1, has_relu=False)
    _ = _ + proj
    return O.relu(_, name='relu' + prefix)


def make_resnet(inpvar, blocks, is_bottleneck=True, output_imm=False, _mid_outputs=None, _outputs=None):
    """
    Build resnet.

    :param src: input tensor, of data type NHWC.
    :param blocks: number of residual blocks for each residual module. Length should be 4.
    :param is_bottleneck: use the bottleneck module or not (1x1 -> 3x3 -> 1x1).
    :param output_imm: whether to output immediate results (conv1 ~ conv5) or not. If true, return value would be `gap, [conv1, conv2, conv3, conv4, conv5]`.
    """


    f = create_bn_relu("conv1", inpvar, 64, 7, stride=2)
    convs_imm = [f]
    f = O.pooling2d("pool1", f, 3, stride=2, padding='SAME', method="MAX")

    pre = [2, 3, 4, 5]
    if _mid_outputs is None:
        _mid_outputs = [64, 128, 256, 512]
    if _outputs is None:
        _outputs = [256, 512, 1024, 2048]
    enable_stride = [False, True, True, True]

    for p, s, mo, o, es in zip(pre, blocks, _mid_outputs, _outputs, enable_stride):
        for i in range(s):
            prefix = "{}_{}".format(p, str(i))
            stride = 1 if not es or i > 0 else 2
            has_proj = False if i > 0 else True
            if is_bottleneck:
                f = create_bottleneck(prefix, f, stride, mo, o, has_proj)
            else:
                f = create_block(prefix, f, stride, mo, o, has_proj)
            print("{}\t{}".format(f.name, f.static_shape))
            convs_imm.append(f)

    f = layer.global_avg_pooling2d('pool5', f)

    if output_imm:
        return f, convs_imm
    else:
        return f


make_resnet_18 = functools.partial(make_resnet, blocks=[2, 2, 2, 2], is_bottleneck=False)
make_resnet_34 = functools.partial(make_resnet, blocks=[3, 4, 6, 3], is_bottleneck=False)
make_resnet_50 = functools.partial(make_resnet, blocks=[3, 4, 6, 3], is_bottleneck=True)
make_resnet_101 = functools.partial(make_resnet, blocks=[3, 4, 23, 3], is_bottleneck=True)
make_resnet_152 = functools.partial(make_resnet, blocks=[3, 8, 36, 3], is_bottleneck=True)

make_resnet_101_xiangyu = functools.partial(make_resnet, blocks=[3, 4, 23, 3], is_bottleneck=True, 
        _outputs=[192, 384, 1024, 2048])


if __name__ == '__main__':
    img = O.placeholder('img', shape=(512, 224, 224, 3))
    out = make_resnet_50(img)

    from tartist.nn import get_default_env
    for i in get_default_env().graph.get_operations():
        print(i.name, i.type, sep='@', end=': ')
        print(*[x.name for x in i.inputs], sep=', ')

