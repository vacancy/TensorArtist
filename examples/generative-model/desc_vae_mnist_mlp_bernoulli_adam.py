# -*- coding:utf8 -*-
# File   : desc_vae_mnist_mlp_bernoulli_adam.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/17/17
# 
# This file is part of TensorArtist

from tartist.core import get_env, get_logger
from tartist.core.utils.naming import get_dump_directory, get_data_directory
from tartist.nn import opr as O, optimizer, summary

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
    'inference': {
        'batch_size': 256,
        'epoch_size': 40
    }
}


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
                if env.phase is env.Phase.TRAIN:
                    with tf.variable_scope('encoder'):
                        _ = x
                        _ = O.fc('fc1', _, 500, nonlin=O.tanh)
                        _ = O.fc('fc2', _, 500, nonlin=O.tanh)
                        mu = O.fc('fc3_mu', _, code_length)
                        log_var = O.fc('fc3_sigma', _, code_length)
                        var = O.exp(log_var)
                        std = O.sqrt(var)
                        epsilon = O.as_varnode(tf.random_normal(O.canonize_sym_shape([x.shape[0], code_length])))
                        z_given_x = mu + std * epsilon
                else:
                    z_given_x = O.as_varnode(tf.random_normal([1, code_length]))

                with tf.variable_scope('decoder'):
                    _ = z_given_x
                    _ = O.fc('fc1', _, 500, nonlin=O.tanh)
                    _ = O.fc('fc2', _, 500, nonlin=O.tanh)
                    _ = O.fc('fc3', _, 784, nonlin=O.sigmoid)
                    _ = _.reshape(-1, h, w, c)
                    x_given_z = _

                if env.phase is env.Phase.TRAIN:
                    with tf.variable_scope('loss'):
                        content_loss = O.raw_cross_entropy_prob('raw_content', x_given_z.flatten2(), x.flatten2())
                        content_loss = content_loss.sum(axis=1).mean(name='content')
                        # distrib_loss = 0.5 * (O.sqr(mu) + O.sqr(std) - 2. * O.log(std + 1e-8) - 1.0).sum(axis=1)
                        distrib_loss = -0.5 * (1. + log_var - O.sqr(mu) - var).sum(axis=1)
                        distrib_loss = distrib_loss.mean(name='distrib')

                        loss = content_loss + distrib_loss
                    dpc.add_output(loss, name='loss', reduce_method='sum')

                dpc.add_output(x_given_z, name='output')

            dpc.set_input_maker(inputs).set_forward_func(forward)

        net.add_all_dpc_outputs(dpc, loss_name='loss')

        if env.phase is env.Phase.TRAIN:
            summary.inference.scalar('loss', net.loss)


def make_optimizer(env):
    wrapper = optimizer.OptimizerWrapper()
    wrapper.set_base_optimizer(optimizer.base.AdamOptimizer(get_env('trainer.learning_rate')))
    wrapper.append_grad_modifier(optimizer.grad_modifier.LearningRateMultiplier([
        ('*/b', 2.0),
    ]))
    # wrapper.append_grad_modifier(optimizer.grad_modifier.WeightDecay([
    #     ('*/W', 0.0005)
    # ]))
    env.set_optimizer(wrapper)


from data_provider_vae_mnist import *


def main_train(trainer):
    from tartist.plugins.trainer_enhancer import summary
    summary.enable_summary_history(trainer)
    summary.enable_echo_summary_scalar(trainer)
    summary.set_error_summary_key(trainer, 'error')

    from tartist.plugins.trainer_enhancer import progress
    progress.enable_epoch_progress(trainer)

    from tartist.plugins.trainer_enhancer import snapshot
    snapshot.enable_snapshot_saver(trainer)

    from tartist.plugins.trainer_enhancer import inference
    inference.enable_inference_runner(trainer, make_dataflow_inference)

    trainer.train()

