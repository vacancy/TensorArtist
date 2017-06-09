# -*- coding:utf8 -*-
# File   : deep_dream.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/16/17
# 
# This file is part of TensorArtist.

import argparse
import os

import numpy as np

from tartist import image
from tartist.core import io
from tartist.core import load_env, get_env, get_logger
from tartist.core.utils.cli import parse_devices
from tartist.nn import Env
from tartist.nn import opr as O

logger = get_logger(__file__)

parser = argparse.ArgumentParser()
parser.add_argument('-w', dest='weight_path', required=True, help='weight path')
parser.add_argument('-i', dest='image_path', required=True, help='input image path')
parser.add_argument('-o', dest='output_path', required=True, help='output directory')
parser.add_argument('-e', '--end', dest='end', default='conv4_3', help='end')
parser.add_argument('-d', '--dev', dest='device', type=str, default='gpu0', help='device id, can be 0/gpu0/cpu0')
parser.add_argument('--iter', dest='nr_iters', type=int, default=1000, help='number of iterations')
parser.add_argument('--save-step', dest='save_step', type=int, default=50, help='save step (in iteration)')
args = parser.parse_args()

args.device = parse_devices([args.device])[0]

__envs__ = {
    'deep_dream': {
        'jitter': 32,
        'learning_rate': 1.5,
        'image_mean': np.array([104, 117, 123], dtype='float32')
    }
}
load_env(__envs__)


def make_network(env, end, h, w):
    with env.create_network() as net:
        img = O.variable('img', np.zeros([1, h, w, 3]))
        net.add_output(img, name='img')

        _ = img
        _ = _ - get_env('deep_dream.image_mean').reshape(1, 1, 1, 3)
        _ = O.pad_rb_multiple_of(_, 32)

        def stacked_conv(prefix, nr_convs, in_, channel, kernel=(3, 3), padding='SAME', nonlin=O.relu):
            for i in range(1, nr_convs + 1):
                in_ = O.conv2d('{}_{}'.format(prefix, i), in_, channel, kernel, padding=padding, nonlin=nonlin)
            return in_

        _ = stacked_conv('conv1', 2, _, 64)
        _ = O.pooling2d('pool1', _, (2, 2))
        _ = stacked_conv('conv2', 2, _, 128)
        _ = O.pooling2d('pool2', _, (2, 2))
        _ = stacked_conv('conv3', 3, _, 256)
        _ = O.pooling2d('pool3', _, (2, 2))
        _ = stacked_conv('conv4', 3, _, 512)
        _ = O.pooling2d('pool4', _, (2, 2))
        _ = stacked_conv('conv5', 3, _, 512)
        _ = O.pooling2d('pool5', _, (2, 2))

        net.add_output(net.find_var_by_name(end + '/bias'), name='end')


def make_step(net):
    """iter only one step, providing end"""

    imgvar = net.outputs['img']
    target = net.outputs['end']
    netin = imgvar

    # random draw ox, oy
    jitter = get_env('deep_dream.jitter')
    ox, oy = np.random.randint(-jitter, jitter+1, 2)

    img = netin.get_value()
    img = np.roll(np.roll(img, ox, 2), oy, 1)  # apply jitter shift

    # compute the gradient
    # one shuold note that we are actually use L2 loss for an activation map to
    # to compute the gradient for the input
    netin.set_value(img)
    loss = 0.5 * (target ** 2.).mean()
    grad = O.grad(loss, imgvar)
    grad = grad.eval()

    # apply gradient ascent, with normalized gradient
    img += get_env('deep_dream.learning_rate') / np.abs(grad).mean() * grad
    img = np.clip(img, 0, 255)

    img = np.roll(np.roll(img, -ox, 2), -oy, 1)  # unshift image
    netin.set_value(img)


def main():
    from tartist.plugins.trainer_enhancer import snapshot

    # read the image, and load the network
    img = image.imread(args.image_path)
    h, w = img.shape[0:2]

    def as_netin(x):
        x = x[np.newaxis, :]
        return np.ascontiguousarray(x, dtype='float32')

    def from_netin(x):
        x = x[0]
        x = np.clip(x, 0, 255.)
        return np.ascontiguousarray(x, dtype='uint8')

    env = Env(Env.Phase.TEST, args.device)
    with env.as_default():
        make_network(env, args.end, h, w)
        snapshot.load_weights_file(env, args.weight_path)

        net = env.network
        netin = net.outputs['img']
        netin.set_value(as_netin(img))

        io.makedir(args.output_path)
        for i in range(args.nr_iters+1):
            if i != 0:
                make_step(net)

            if i % args.save_step == 0:
                output = from_netin(netin.get_value())

                output_path = os.path.join(args.output_path, 'iter_{:04d}.jpg'.format(i))
                image.imwrite(output_path, output)
                logger.critical('Output written: {}'.format(output_path))


if __name__ == '__main__':
   main()
