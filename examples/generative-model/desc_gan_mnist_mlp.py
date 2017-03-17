# -*- coding:utf8 -*-
# File   : desc_gan_mnist_mlp.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/17/17
# 
# This file is part of TensorArtist

from tartist.core import get_env, get_logger
from tartist.core.utils.naming import get_dump_directory, get_data_directory
from tartist.nn import opr as O, optimizer, summary, train

import tensorflow as tf

logger = get_logger(__file__)

__envs__ = {
    'dir': {
        'root': get_dump_directory(__file__),
        'data': get_data_directory('WellKnown/mnist')
    },

    'trainer': {
        'learning_rate': 0.001,

        'batch_size': 100,
        'epoch_size': 500,
        'nr_epochs': 100,

        'env_flags': {
            'log_device_placement': False
        }
    },
}

__trainer_cls__ = train.gan.GANTrainer
__trainer_env_cls__ = train.gan.GANTrainerEnv


def make_network(env):
    with env.create_network() as net:
        code_length = 20
        h, w, c = 28, 28, 1

        dpc = env.create_dpcontroller()
        with dpc.activate():
            def inputs():
                img = O.placeholder('img', shape=(None, h, w, c))
                return [img]

            def forward(x):
                z = O.as_varnode(tf.random_normal([1, code_length]))
                with tf.variable_scope('generator'):
                    out = None

                with tf.variable_scope('discriminator', reuse=True):
                    pass

                if env.phase is env.Phase.TRAIN:
                    with tf.variable_scope('discriminator'):
                        pass

                if env.phase is env.Phase.TRAIN:
                    g_loss = None
                    d_loss = None

                    dpc.add_output(g_loss, name='g_loss', reduce_method='sum')
                    dpc.add_output(d_loss, name='d_loss', reduce_method='sum')

                dpc.add_output(out, name='output')

            dpc.set_input_maker(inputs).set_forward_func(forward)

        net.add_all_dpc_outputs(dpc)


def make_optimizer(env):
    lr = optimizer.base.make_optimizer_variable('learning_rate', get_env('trainer.learning_rate'))

    with tf.variable_scope('generator'):
        wrapper = optimizer.OptimizerWrapper()
        wrapper.set_base_optimizer(optimizer.base.AdamOptimizer(lr))
        wrapper.append_grad_modifier(optimizer.grad_modifier.LearningRateMultiplier([
            ('*/b', 2.0),
        ]))
        env.set_g_optimizer(wrapper)

    with tf.variable_scope('discriminator'):
        wrapper = optimizer.OptimizerWrapper()
        wrapper.set_base_optimizer(optimizer.base.AdamOptimizer(lr))
        wrapper.append_grad_modifier(optimizer.grad_modifier.LearningRateMultiplier([
            ('*/b', 2.0),
        ]))
        env.set_d_optimizer(wrapper)


from data_provider_gan_mnist import *


def main_train(trainer):
    from tartist.plugins.trainer_enhancer import summary
    summary.enable_summary_history(trainer)
    summary.enable_echo_summary_scalar(trainer)
    summary.set_error_summary_key(trainer, 'error')

    from tartist.plugins.trainer_enhancer import progress
    progress.enable_epoch_progress(trainer)

    from tartist.plugins.trainer_enhancer import snapshot
    snapshot.enable_snapshot_saver(trainer)

    # TODO(MJY): does not support inference_runner now
    # from tartist.plugins.trainer_enhancer import inference
    # inference.enable_inference_runner(trainer, make_dataflow_inference)

    trainer.train()

