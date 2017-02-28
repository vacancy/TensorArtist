# -*- coding:utf8 -*-
# File   : desc_cifar.py
# Author : Jiayuan Mao
#          Honghua Dong
# Email  : maojiayuan@gmail.com
#          dhh19951@gmail,com
# Date   : 2/27/17
#
# This file is part of TensorArtist

import tensorflow as tf

from tartist.core import get_env, get_logger
from tartist.core.utils.naming import get_dump_directory, get_data_directory
from tartist.nn import opr as O, optimizer, summary

import functools

logger = get_logger(__file__)

__envs__ = {
    'dir': {
        'root': get_dump_directory(__file__),
        'data': get_data_directory('WellKnown/cifar')
    },

    'dataset': {
        'nr_classes' : 10
    },

    'trainer': {
        'epoch_size': 128,
        'nr_iters': 1280,
        'learning_rate': 0.01,
        'batch_size': 64,

        'env_flags': {
            'log_device_placement': False
        }
    }

}

def make_network(env):
    with env.create_network() as net:
        nr_classes = get_env('dataset.nr_classes')

        conv_bn_relu = functools.partial(O.conv2d, nonlin=O.bn_relu)
        conv2d = conv_bn_relu 

        dpc = env.create_dpcontroller()
        with dpc.activate():
            def inputs():
                h, w, c = 32, 32, 3
                img = O.placeholder('img', shape=(None, h, w, c))
                return [img]

            def forward(img):
                _ = img
                _ = conv2d('conv1.1', _, 16, (3, 3), padding='SAME')
                _ = conv2d('conv1.2', _, 16, (3, 3), padding='SAME')
                _ = O.pooling2d('pool1', _, kernel=3, stride=2)
                _ = conv2d('conv2.1', _, 32, (3, 3), padding='SAME')
                _ = conv2d('conv2.2', _, 32, (3, 3), padding='SAME')
                _ = O.pooling2d('pool2', _, kernel=3, stride=2)
                _ = conv2d('conv3.1', _, 64, (3, 3), padding='VALID')
                _ = conv2d('conv3.2', _, 64, (3, 3), padding='VALID')
                _ = conv2d('conv3.3', _, 64, (3, 3), padding='VALID')

                dpc.add_output(_, name='feature')

            dpc.set_input_maker(inputs).set_forward_func(forward)

        _ = dpc.outputs['feature']
        _ = O.fc('fc1', _, 128, nonlin=O.relu)
        _ = O.fc('fc2', _, 64, nonlin=O.relu)
        _ = O.fc('linear', _, nr_classes)

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
            summary.scalar('error', 1. - accuracy)


def make_optimizer(env):
    wrapper = optimizer.OptimizerWrapper()
    wrapper.set_base_optimizer(optimizer.base.MomentumOptimizer(get_env('trainer.learning_rate'), 0.9))
    wrapper.append_grad_modifier(optimizer.grad_modifier.LearningRateMultiplier([
        ('*/b', 2.0),
    ]))
    env.set_optimizer(wrapper)

from data_provider import make_dataflow_train as make_dataflow


def main_train(trainer):
    from tartist.plugins.trainer_enhancer import summary_logger
    summary_logger.enable_summary_history(trainer)
    summary_logger.enable_echo_summary_scalar(trainer)
    summary_logger.set_error_summary_key(trainer, 'error')

    from tartist.plugins.trainer_enhancer import progress
    progress.enable_epoch_progress(trainer)

    from tartist.plugins.trainer_enhancer import snapshot
    snapshot.enable_snapshot_saver(trainer)

    trainer.train()

