# -*- coding:utf8 -*-
# File   : desc_mnist.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/30/16
#
# This file is part of TensorArtist

import os.path as osp
import tensorflow as tf
import numpy as np
import numpy.random as npr

from tartist.core import get_env
from tartist.core.utils.naming import get_dump_directory, get_data_directory
from tartist.nn import Env, opr as O, optimizer, summary
from tartist.data import flow

__envs__ = {
    'dir': {
        'root': get_dump_directory(__file__),
        'data': get_data_directory('WellKnown/mnist')
    },

    'trainer': {
        'nr_iters': 1000,
        'learning_rate': 0.1,
        'batch_size': 64,

        'env_flags': {
            'log_device_placement': False
        }
    }
}

def make_network(env):
    with env.create_network() as net:

        dpc = env.create_dpcontroller()
        with dpc.activate():
            def inputs():
                h, w, c = 28, 28, 1
                img = O.placeholder('img', shape=(None, h, w, c))
                return [img]

            def forward(img):
                _ = img
                _ = O.conv2d('conv1', _, 16, (3, 3), padding='SAME', nonlin=O.identity)
                _ = O.batchnorm('bn1', _)
                _ = O.relu(_)
                _ = O.pooling2d('pool1', _, kernel=2)
                _ = O.conv2d('conv2', _, 32, (3, 3), padding='SAME', nonlin=O.identity)
                _ = O.batchnorm('bn2', _)
                _ = O.relu(_)
                _ = O.pooling2d('pool2', _, kernel=2)
                dpc.add_output(_, name='feature')
                summary.scalar('feature/rms', _.rms())

            dpc.set_input_maker(inputs).set_forward_func(forward)

        _ = dpc.outputs['feature']
        _ = O.fc('fc1', _, 64)
        _ = O.fc('fc2', _, 10)

        # it's safe to use tf.xxx and O.xx together
        prob = O.softmax(_, name='prob')
        pred = _.argmax(axis=1, name='pred').astype(tf.int32)
        net.add_output(prob)
        net.add_output(pred)

        if env.phase is env.Phase.TRAIN:
            label = O.placeholder('label', shape=(None, ), dtype=tf.int32)
            loss = O.sparse_softmax_cross_entropy_with_logits(logits=_, labels=label).mean()
            loss = O.identity(loss, name='loss')
            net.set_loss(loss)

            accuracy = O.eq(label, pred).astype('float32').mean()
            summary.scalar('accuracy', accuracy)


def make_optimizer(env):
    wrapper = optimizer.OptimizerWrapper()
    wrapper.set_base_optimizer(optimizer.base.SGDOptimizer(get_env('trainer.learning_rate')))
    wrapper.append_grad_modifier(optimizer.grad_modifier.LearningRateMultiplier([
        ('*/b', 2.0),
    ]))
    env.set_optimizer(wrapper)

from data_provider import make_dataflow_train as make_dataflow

# if True:
#     f = env.make_func()
#     f.compile(env.network.outputs['pred'])
#     logger.info('final pred  ={}'.format(f(img=data_fake['img'])))
#     logger.info('ground truth={}'.format(data_fake['label']))

