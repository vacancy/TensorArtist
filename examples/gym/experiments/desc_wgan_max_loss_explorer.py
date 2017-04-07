# -*- coding:utf8 -*-
# File   : desc_wgan_max_loss_explorer.py
# Author : Jiayuan Mao
#          Honghua Dong
# Email  : maojiayuan@gmail.com
#          dhh19951@gmail.com
# Date   : 4/7/17
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
        'input_shape': (84, 84),
        'frame_history': 4,
        'limit_length': 40000
    },
    'trainer': {
        'learning_rate': 2e-4,

        'batch_size': 128,
        'epoch_size': 1000,
        'nr_epochs': 200,

        'env_flags': {
            'log_device_placement': False
        }
    }
}

__trainer_cls__ = train.gan.GANTrainer
__trainer_env_cls__ = train.gan.GANTrainerEnv

from data_provider_gan_gym import *


def make_network(env):
    with env.create_network() as net:
        h, w, c = get_input_shape()
        z_dim = 100

        dpc = env.create_dpcontroller()
        with dpc.activate():
            def inputs():
                state = O.placeholder('state', shape=(None, h, w, c))
                action = O.placeholder('action', shape=(None, ), dtype=tf.int64)
                if env.phase is env.Phase.TRAIN:
                    state_real = O.placeholder('next_state', shape=(None, h, w, 3))
                    return [state, action, state_real]
                else:
                    return [state, action]

            def feature(x):
                _ = x
                with O.argscope(O.conv2d, nonlin=O.relu):
                    _ = O.conv2d('conv0', _, 32, 5)
                    _ = O.max_pooling2d('pool0', _, 2)
                    _ = O.conv2d('conv1', _, 32, 5)
                    _ = O.max_pooling2d('pool1', _, 2)
                    _ = O.conv2d('conv2', _, 64, 4)
                    _ = O.max_pooling2d('pool2', _, 2)
                    _ = O.conv2d('conv3', _, 64, 3)
                    _ = O.fc('fc0', _, 512, nonlin=O.p_relu)
                return _


            def generator(state, action, z):
                w_init = tf.truncated_normal_initializer(stddev=0.02)
                with O.argscope(O.conv2d, O.deconv2d, kernel=4, stride=2, W=w_init),\
                     O.argscope(O.fc, W=w_init):

                    a = O.one_hot(action, get_player_nr_actions())
                    _ = feature(state)
                    _ = O.concat([a, _, z], axis=1)
                    _ = O.fc('fc1', _, 1024, nonlin=O.bn_relu)
                    _ = O.fc('fc2', _, 128 * 7 * 7, nonlin=O.bn_relu)
                    _ = O.reshape(_, [-1, 7, 7, 128])
                    _ = O.deconv2d('deconv1', _, 64, kernel=5, stride=3, nonlin=O.bn_relu)
                    _ = O.deconv2d('deconv2', _, 32, nonlin=O.bn_relu)
                    _ = O.deconv2d('deconv3', _, 3)
                    #_ = O.tanh(_, name='out')
                return _

            def discriminator(state, action, pred):
                w_init = tf.truncated_normal_initializer(stddev=0.02)
                with O.argscope(O.conv2d, O.deconv2d, kernel=4, stride=2, W=w_init),\
                     O.argscope(O.fc, W=w_init),\
                     O.argscope(O.leaky_relu, alpha=0.2):

                    _ = O.concat([state, pred], axis=3) #84, 84, 15
                    _ = O.conv2d('conv1', _, 32, nonlin=O.leaky_relu)
                    _ = O.conv2d('conv2', _, 64, nonlin=O.bn_nonlin)
                    _ = O.leaky_relu(_)
                    _ = O.conv2d('conv3', _, 128, kernel=5, stride=3, nonlin=O.bn_nonlin)
                    _ = O.leaky_relu(_)
                    _ = O.fc('fc1', _, 1024, nonlin=O.bn_nonlin)
                    _ = O.leaky_relu(_)
                    a = O.one_hot(action, get_player_nr_actions())
                    _ = O.concat([_, a], axis=1)
                    _ = O.fc('fca', _, 128, nonlin=O.bn_nonlin)
                    _ = O.leaky_relu(_)
                    _ = O.fc('fct', _, 1)

                return _

            def forward(state, action, state_real=None):
                g_batch_size = get_env('trainer.batch_size') if env.phase is env.Phase.TRAIN else 1
                z = O.random_normal([g_batch_size, z_dim])

                with tf.variable_scope(GANGraphKeys.GENERATOR_VARIABLES):
                    state_pred = O.split(state, c // 3, axis=3)[-1] + generator(state, action, z)
                state_pred = O.min(O.max(state_pred, 0), 255)
                # tf.summary.image('generated-samples', state_pred, max_outputs=30)

                with tf.variable_scope(GANGraphKeys.DISCRIMINATOR_VARIABLES):
                    logits_fake = discriminator(state, action, state_pred)
                dpc.add_output(state_pred, name='output')

                if env.phase is env.Phase.TRAIN:
                    with tf.variable_scope(GANGraphKeys.DISCRIMINATOR_VARIABLES, reuse=True):
                        logits_real = discriminator(state, action, state_real)

                    d_loss = (logits_fake - logits_real).mean()
                    g_loss = -logits_fake.mean()
                    dpc.add_output(d_loss, name='d_loss', reduce_method='sum')
                    dpc.add_output(g_loss, name='g_loss', reduce_method='sum')
                

            dpc.set_input_maker(inputs).set_forward_func(forward)

        net.add_all_dpc_outputs(dpc)


def make_optimizer(env):
    lr = optimizer.base.make_optimizer_variable('learning_rate', get_env('trainer.learning_rate'))

    wrapper = optimizer.OptimizerWrapper()
    wrapper.set_base_optimizer(optimizer.base.RMSPropOptimizer(lr))
    #wrapper.append_grad_modifier(optimizer.grad_modifier.LearningRateMultiplier([
    #    ('*/b', 2.0),
    #]))
    env.set_g_optimizer(wrapper)
    env.set_d_optimizer(wrapper)


def enable_clip_param(trainer):
    # apply clip on params of discriminator
    limit = get_env('trainer.clip_limit', 0.01)
    ops = []
    with trainer.env.as_default():
        sess = trainer.env.session
        var_list = tf.trainable_variables()
        for v in var_list:
            if v.name.startswith(GANGraphKeys.DISCRIMINATOR_VARIABLES + '/'):
                ops.append(v.assign(tf.clip_by_value(v, -limit, limit)))
    op = tf.group(*ops)
    
    def do_clip_params(trainer, inp, out):
        trainer.env.session.run(op)
    
    from tartist.core import register_event
    register_event(trainer, 'iter:after', do_clip_params)


def main_train(trainer):
    from tartist.plugins.trainer_enhancer import summary
    summary.enable_summary_history(trainer)
    summary.enable_echo_summary_scalar(trainer)

    from tartist.plugins.trainer_enhancer import progress
    progress.enable_epoch_progress(trainer)

    from tartist.plugins.trainer_enhancer import snapshot
    snapshot.enable_snapshot_saver(trainer)

    enable_clip_param(trainer)


    trainer.train()

