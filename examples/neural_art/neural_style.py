# -*- coding:utf8 -*-
# File   : neural_style.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/27/17
# 
# This file is part of TensorArtist


from tartist.core import load_env, get_env, get_logger
from tartist.core import io
from tartist.core.utils.cli import parse_devices
from tartist.nn import Env
from tartist.nn import opr as O, optimizer
from tartist.nn.train import TrainerEnv
import nart_opr

import argparse
import collections
import os
import cv2
import numpy as np

logger = get_logger(__file__)

__envs__ = {
    'neural_style': {
        'content_layers': [('conv4_2', 1.)],
        'style_layers': [('conv1_1', 1.), ('conv2_1', 1.5), ('conv3_1', 2.), ('conv4_1', 2.5), ('conv5_1', 3.)],
        'style_strenth': 500.,
        'image_noise_ratio': 0.7,
        'image_mean': np.array([104, 117, 123], dtype='float32'),
        'learning_rate': 2.0
    }
}

load_env(__envs__)

parser = argparse.ArgumentParser()
parser.add_argument('-w', dest='weight_path', required=True, help='weight path')
parser.add_argument('-i', dest='image_path', required=True, help='input image path')
parser.add_argument('-s', dest='style_path', required=True, help='style image path')
parser.add_argument('-o', dest='output_path', required=True, help='output directory')
parser.add_argument('-d', '--dev', dest='device', type=str, default='gpu0', help='device id, can be 0/gpu0/cpu0')
parser.add_argument('--iter', dest='nr_iters', type=int, default=1000, help='number of iterations')
parser.add_argument('--save-step', dest='save_step', type=int, default=50, help='save step (in iteration)')
args = parser.parse_args()

args.device = parse_devices([args.device])[0]


def make_network(env, h=None, w=None):
    with env.create_network() as net:
        if h is None:
            img = O.placeholder('img', shape=(1, None, None, 3))
        else:
            img = O.variable('img', np.zeros([1, h, w, 3]))
        net.add_output(img, name='img')

        _ = img
        _ = _ - get_env('neural_style.image_mean').reshape(1, 1, 1, 3)
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

        for l in get_env('neural_style.content_layers'):
            net.add_output(net.find_var_by_name(l[0]), name=l[0])
        for l in get_env('neural_style.style_layers'):
            net.add_output(net.find_var_by_name(l[0]), name=l[0])


def main():
    from tartist.plugins.trainer_enhancer import snapshot

    img = cv2.imread(args.image_path)
    smg = cv2.imread(args.style_path)
    h, w = img.shape[0:2]

    env = Env(master_dev=args.device)
    with env.as_default():
        make_network(env)
        snapshot.load_weights_file(env, args.weight_path)

        func = env.make_func()
        func.compile(env.network.outputs)
    
        env.initialize_all_variables()
        res_img = func(img=img[np.newaxis])
        res_smg = func(img=img[np.newaxis])
    
    # create a new env for train
    env = TrainerEnv(master_dev=args.device)
    with env.as_default():
        make_network(env, h, w)
        snapshot.load_weights_file(env, args.weight_path)

    net = env.network
    netin = net.outputs['img'].taop

    with env.as_default():
        outputs = net.outputs
        loss_content = 0.
        for i in get_env('neural_style.content_layers'):
            loss_content += i[1] * nart_opr.get_content_loss(res_img[i[0]], outputs[i[0]])
        loss_style = 0.
        for i in get_env('neural_style.style_layers'):
            loss_style += i[1] * nart_opr.get_style_loss(res_smg[i[0]], outputs[i[0]])
        loss = loss_content + loss_style * get_env('neural_style.style_strenth')
        net.set_loss(loss)

    wrapper = optimizer.OptimizerWrapper()
    wrapper.set_base_optimizer(optimizer.base.AdamOptimizer(get_env('neural_style.learning_rate')))
    wrapper.append_grad_modifier(optimizer.grad_modifier.LearningRateMultiplier([
        ('img', 1.0),
        ('*', 0.0),
    ]))
    env.set_optimizer(wrapper)

    func = env.make_optimizable_func()
    func.compile({'loss': net.loss})
    env.initialize_all_variables()

    noise_img = np.random.uniform(-20, 20, (h, w, 3)).astype('float32')
    image_mean = get_env('neural_style.image_mean')
    image_noise_ratio = get_env('neural_style.image_noise_ratio')
    jmg = (1 - image_noise_ratio) * (img - image_mean) + image_noise_ratio * noise_img + image_mean
    netin.set_value(jmg[np.newaxis])

    io.makedir(args.output_path)

    for i in range(args.nr_iters + 1):
        if i != 0:
            r = func()
            logger.info('iter {}: loss = {}'.format(i, float(r['loss'])))

        if i % args.save_step == 0:
            output = netin.get_value()[0]

            output_path = os.path.join(args.output_path, 'iter_{:04d}.jpg'.format(i))
            cv2.imwrite(output_path, output)
            logger.critical('output written: {}'.format(output_path))


if __name__ == '__main__':
    main()

