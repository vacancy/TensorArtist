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
from tartist.nn.train.gan import GANGraphKeys

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
        is_train = env.phase == env.Phase.TRAIN

        dpc = env.create_dpcontroller()
        with dpc.activate():
            def inputs():
                img = O.placeholder('img', shape=(None, h, w, c))
                return [img]

            def forward(img):
                g_batch_size = get_env('trainer.batch_size') if env.phase is env.Phase.TRAIN else 1
                z = O.as_varnode(tf.random_normal([g_batch_size, code_length]))
                with tf.variable_scope(GANGraphKeys.GENERATOR_VARIABLES):
                    _ = z
                    with O.argscope(O.fc, nonlin=O.tanh):
                        _ = O.fc('fc1', _, 500)
                    _ = O.fc('fc3', _, 784, nonlin=O.sigmoid)
                    x_given_z = _.reshape(-1, 28, 28, 1)

                def discriminator(x):
                    _ = x
                    with O.argscope(O.fc, nonlin=O.tanh):
                        _ = O.fc('fc1', _, 500)
                    _ = O.fc('fc3', _, 1)
                    logits = _
                    return logits

                if is_train:
                    with tf.variable_scope(GANGraphKeys.DISCRIMINATOR_VARIABLES):
                        logits_real = discriminator(img).flatten()
                        score_real = O.sigmoid(logits_real)

                with tf.variable_scope(GANGraphKeys.DISCRIMINATOR_VARIABLES, reuse=is_train):
                    logits_fake = discriminator(x_given_z).flatten()
                    score_fake = O.sigmoid(logits_fake)

                if is_train:
                    with tf.variable_scope('loss'):
                        d_loss = (
                            O.sigmoid_cross_entropy_with_logits(
                                labels=O.ones([logits_real.shape[0]]), logits=logits_real) +
                            O.sigmoid_cross_entropy_with_logits(
                                labels=O.zeros([logits_fake.shape[0]]), logits=logits_fake)
                        ).mean() / 2.

                        g_loss = O.sigmoid_cross_entropy_with_logits(
                            labels=O.ones([logits_fake.shape[0]]), logits=logits_fake).mean()

                        accuracy_real = (score_real > 0.5).astype('float32').mean()
                        accuracy_fake = (score_fake < 0.5).astype('float32').mean()

                    dpc.add_output(g_loss, name='g_loss', reduce_method='sum')
                    dpc.add_output(d_loss, name='d_loss', reduce_method='sum')
                    dpc.add_output(accuracy_real, name='d_accuracy_real', reduce_method='sum')
                    dpc.add_output(accuracy_fake, name='d_accuracy_fake', reduce_method='sum')

                dpc.add_output(x_given_z, name='output')
                dpc.add_output(score_fake, name='score')

            dpc.set_input_maker(inputs).set_forward_func(forward)

        if is_train:
            for acc in ['d_accuracy_real', 'd_accuracy_fake']:
                summary.scalar(acc, dpc.outputs[acc], collections=[GANGraphKeys.DISCRIMINATOR_SUMMARIES])

        net.add_all_dpc_outputs(dpc)


def make_optimizer(env):
    lr = optimizer.base.make_optimizer_variable('learning_rate', get_env('trainer.learning_rate'))

    wrapper = optimizer.OptimizerWrapper()
    wrapper.set_base_optimizer(optimizer.base.AdamOptimizer(lr))
    wrapper.append_grad_modifier(optimizer.grad_modifier.LearningRateMultiplier([
        ('*/b', 2.0),
    ]))
    env.set_g_optimizer(wrapper)
    env.set_d_optimizer(wrapper)


from data_provider_gan_mnist import *


def main_train(trainer):
    from tartist.plugins.trainer_enhancer import summary
    summary.enable_summary_history(trainer)
    summary.enable_echo_summary_scalar(trainer)

    from tartist.plugins.trainer_enhancer import progress
    progress.enable_epoch_progress(trainer)

    from tartist.plugins.trainer_enhancer import snapshot
    snapshot.enable_snapshot_saver(trainer)

    # TODO(MJY): does not support inference_runner now
    # from tartist.plugins.trainer_enhancer import inference
    # inference.enable_inference_runner(trainer, make_dataflow_inference)

    trainer.train()
