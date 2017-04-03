# -*- coding:utf8 -*-
# File   : desc_gan_predictor_cnn.py
# Author : Jiayuan Mao
#          Honghua Dong
# Email  : maojiayuan@gmail.com
#          dhh19951@gmail.com
# Date   : 3/29/17
#
# This file is part of TensorArtist

import tensorflow as tf

from tartist.core import get_env, get_logger
from tartist.core.utils.cache import cached_result
from tartist.core.utils.naming import get_dump_directory, get_data_directory
from tartist.nn import opr as O, optimizer, summary, train
from tartist.nn.train.gan import GANGraphKeys

import functools

logger = get_logger(__file__)

__envs__ = {
    'dir': {
        'root': get_dump_directory(__file__)
    },
    'gym': {
        'env_name': 'Breakout-v0',
        'input_shape': (210, 160),
        'frame_history': 4,
        'limit_length': 40000
    },
    'trainer': {
        'learning_rate': 0.0001,

        'batch_size': 64,
        'epoch_size': 1000,
        'nr_epochs': 200,

        'env_flags': {
            'log_device_placement': False
        }
    },
    'inference': {
        'batch_size': 64,
        'epoch_size': 50
    }
}

from data_provider_atari import *


def make_network(env):
    with env.create_network() as net:
        h, w, c = get_input_shape()

        def pad(_, h=0, w=0):
            return O.pad(_, [[0, 0], [h, h], [w, w], [0, 0]], 'CONSTANT')

        def crop(_, h=0, w=0):
            if h > 0:
                _ = _[:, h:-h, :, :]
            if w > 0:
                _ = _[:, :, w:-w, :]
            return _

        dpc = env.create_dpcontroller()
        with dpc.activate():
            def inputs():
                state = O.placeholder('state', shape=(None, h, w, c))
                return [state]

            def forward(state):
                _ = state
                _ = O.conv2d('conv1', pad(_, 0, 1), 64, 8, stride=2, padding='VALID', nonlin=O.relu)
                _ = O.conv2d('conv2', pad(_, 1, 1), 128, 6, stride=2, padding='VALID', nonlin=O.relu)
                _ = O.conv2d('conv3', pad(_, 1, 1), 128, 6, stride=2, padding='VALID', nonlin=O.relu)
                _ = O.conv2d('conv4', _, 128, 4, stride=2, padding='VALID', nonlin=O.relu)
                dpc.add_output(_, name='feature')

            dpc.set_input_maker(inputs).set_forward_func(forward)

        _ = dpc.outputs['feature']
        _ = O.fc('fc-enc1', _, 2048, nonlin=O.relu)
        
        #transfer
        _ = O.fc('fc-enc2', _, 2048)
        action = O.placeholder('action', shape=(None, ), dtype=tf.int64)
        action = O.one_hot(action, get_player_nr_actions())
        _ = _ * O.fc('fc-act', action, 2048)
        _ = O.fc('fc-dec2', _, 2048)

        #decode
        _ = O.fc('fc-dec1', _, 11264, nonlin=O.relu)
        _ = _.reshape(-1, 11, 8, 128)
        _ = O.deconv2d('deconv4', _, 128, 4, stride=2, padding='VALID', nonlin=O.relu)
        _ = crop(O.deconv2d('deconv3', _, 128, 6, stride=2, padding='VALID', nonlin=O.relu), 1, 1)
        _ = crop(O.deconv2d('deconv2', _, 128, 6, stride=2, padding='VALID', nonlin=O.relu), 1, 1)
        _ = crop(O.deconv2d('deconv1', _, 3, 8, stride=2, padding='VALID'), 0, 1)

        net.add_output(_, 'output')

        if env.phase is env.Phase.TRAIN:
            label = O.placeholder('next_state', shape=(None, h, w, 3))
            loss = O.raw_l2_loss('l2_loss', _, label).mean(name='loss')
            net.set_loss(loss)

            summary.inference.scalar('loss', loss)


def make_optimizer(env):
    lr = optimizer.base.make_optimizer_variable('learning_rate', get_env('trainer.learning_rate'))

    wrapper = optimizer.OptimizerWrapper()
    wrapper.set_base_optimizer(optimizer.base.AdamOptimizer(lr))
    wrapper.append_grad_modifier(optimizer.grad_modifier.LearningRateMultiplier([
        ('*/b', 2.0),
    ]))
    env.set_optimizer(wrapper)


def main_train(trainer):
    from tartist.plugins.trainer_enhancer import summary
    summary.enable_summary_history(trainer)
    summary.enable_echo_summary_scalar(trainer)

    from tartist.plugins.trainer_enhancer import progress
    progress.enable_epoch_progress(trainer)

    from tartist.plugins.trainer_enhancer import snapshot
    snapshot.enable_snapshot_saver(trainer)

    from tartist.plugins.trainer_enhancer import inference
    inference.enable_inference_runner(trainer, make_dataflow_inference)

    trainer.train()

