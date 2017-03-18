# -*- coding:utf8 -*-
# File   : desc_GAN.py
# Author : Jiayuan Mao
#          Honghua Dong
# Email  : maojiayuan@gmail.com
#          dhh19951@gmail,com
# Date   : 3/10/17
#
# This file is part of TensorArtist

import tensorflow as tf

from tartist.core import get_env, get_logger
from tartist.core.utils.naming import get_dump_directory, get_data_directory
from tartist.nn import opr as O, optimizer, summary, train
from tartist.nn.train.gan import GANGraphKeys

import functools

logger = get_logger(__file__)

__envs__ = {
    'dir': {
        'root': get_dump_directory(__file__),
        'data': get_data_directory('WellKnown/mnist')
    },

    'dataset': {
        'nr_classes': 10,
        'input_shape': 28,
    },

    'trainer': {
        'learning_rate': 2e-4,

        'batch_size': 128,
        'epoch_size': 390,
        'nr_epochs': 200,

        'env_flags': {
            'log_device_placement': False
        }
    }
}

__trainer_cls__ = train.gan.GANTrainer
__trainer_env_cls__ = train.gan.GANTrainerEnv

def make_network(env):
    with env.create_network() as net:
        h, w, c = 28, 28, 1
        z_dim = 100

        dpc = env.create_dpcontroller()
        with dpc.activate():
            def inputs():
                img = O.placeholder('img', shape=(None, h, w, c))
                return [img]

            def generator(z):
                w_init = tf.truncated_normal_initializer(stddev=0.02)
                with O.argscope(O.conv2d, O.deconv2d, kernel=4, stride=2, W=w_init),\
                     O.argscope(O.fc, W=w_init):

                    _ = z
                    _ = O.fc('fc0', _, 1024, nonlin=O.bn_relu)
                    _ = O.fc('fc1', _, 128 * 7 * 7, nonlin=O.bn_relu)
                    _ = O.reshape(_, [-1, 7, 7, 128])
                    _ = O.deconv2d('deconv1', _, 64, nonlin=O.bn_relu)
                    _ = O.deconv2d('deconv2', _, 1)
                    _ = O.sigmoid(_, name='gen')
                return _

            def discriminator(img):
                w_init = tf.truncated_normal_initializer(stddev=0.02)
                with O.argscope(O.conv2d, O.deconv2d, kernel=4, stride=2, W=w_init),\
                     O.argscope(O.fc, W=w_init),\
                     O.argscope(O.leaky_relu, alpha=0.2):

                    _ = img
                    _ = O.conv2d('conv0', _, 64, nonlin=O.leaky_relu)
                    _ = O.conv2d('conv1', _, 128, nonlin=O.bn_nonlin)
                    _ = O.leaky_relu(_)
                    _ = O.fc('fc2', _, 1024, nonlin=O.bn_nonlin)
                    _ = O.leaky_relu(_)
                    _ = O.fc('fct', _, 1)
                return _

            def forward(x):
                g_batch_size = get_env('trainer.batch_size') if env.phase is env.Phase.TRAIN else 1
                z = O.random_normal([g_batch_size, z_dim])

                with tf.variable_scope(GANGraphKeys.GENERATOR_VARIABLES):
                    img_gen = generator(z)
                #tf.summary.image('generated-samples', img_gen, max_outputs=30)

                with tf.variable_scope(GANGraphKeys.DISCRIMINATOR_VARIABLES):
                    logits_fake = discriminator(img_gen)
                    score_fake = O.sigmoid(logits_fake)
                dpc.add_output(img_gen, name='output')
                dpc.add_output(score_fake, name='score')

                if env.phase is env.Phase.TRAIN:
                    with tf.variable_scope(GANGraphKeys.DISCRIMINATOR_VARIABLES, reuse=True):
                        logits_real = discriminator(x)
                        score_real = O.sigmoid(logits_real)
                    #build loss
                    with tf.variable_scope('loss'):
                        d_loss_real = O.sigmoid_cross_entropy_with_logits(
                            logits=logits_real, labels=O.ones_like(logits_real)).mean()
                        d_loss_fake = O.sigmoid_cross_entropy_with_logits(
                            logits=logits_fake, labels=O.zeros_like(logits_fake)).mean()
                        g_loss = O.sigmoid_cross_entropy_with_logits(
                            logits=logits_fake, labels=O.ones_like(logits_fake)).mean()

                    d_acc_real = (score_real > 0.5).astype('float32').mean()
                    d_acc_fake = (score_fake < 0.5).astype('float32').mean()
                    g_accuracy = (score_fake > 0.5).astype('float32').mean()

                    d_accuracy = .5 * (d_acc_real + d_acc_fake)
                    d_loss = .5 * (d_loss_real + d_loss_fake)

                    dpc.add_output(d_loss, name='d_loss', reduce_method='sum')
                    dpc.add_output(d_accuracy, name='d_accuracy', reduce_method='sum')
                    dpc.add_output(d_acc_real, name='d_acc_real', reduce_method='sum')
                    dpc.add_output(d_acc_fake, name='d_acc_fake', reduce_method='sum')
                    dpc.add_output(g_loss, name='g_loss', reduce_method='sum')
                    dpc.add_output(g_accuracy, name='g_accuracy', reduce_method='sum')

            dpc.set_input_maker(inputs).set_forward_func(forward)

        if env.phase is env.Phase.TRAIN:
            for acc in ['d_accuracy', 'd_acc_real', 'd_acc_fake']:
                summary.scalar(acc, dpc.outputs[acc], collections=[GANGraphKeys.DISCRIMINATOR_SUMMARIES])
            summary.scalar('g_accuracy', dpc.outputs['g_accuracy'], collections=[GANGraphKeys.GENERATOR_SUMMARIES])

        net.add_all_dpc_outputs(dpc)


def make_optimizer(env):
    lr = optimizer.base.make_optimizer_variable('learning_rate', get_env('trainer.learning_rate'))

    wrapper = optimizer.OptimizerWrapper()
    wrapper.set_base_optimizer(optimizer.base.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3))
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

    trainer.train()

