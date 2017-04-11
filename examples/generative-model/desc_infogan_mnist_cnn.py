# -*- coding:utf8 -*-
# File   : desc_infogan_mnist_cnn.py
# Author : Jiayuan Mao
#          Honghua Dong
# Email  : maojiayuan@gmail.com
#          dhh19951@gmail,com
# Date   : 3/17/17
#
# This file is part of TensorArtist

import numpy as np
import tensorflow as tf

from tartist.app import gan
from tartist.app.gan import GANGraphKeys
from tartist.core import get_env, get_logger
from tartist.core.utils.naming import get_dump_directory, get_data_directory
from tartist.nn import opr as O, optimizer, summary

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
    },

    'demo': {
        'customized': True,
        'mode': 'infogan',
        'infogan': {
            'grid_desc': ('5h', '2v', '5h', '5v')
        }
    }
}

__trainer_cls__ = gan.GANTrainer
__trainer_env_cls__ = gan.GANTrainerEnv


def make_network(env):
    with env.create_network() as net:
        g_batch_size = get_env('trainer.batch_size') if env.phase is env.Phase.TRAIN else 1
        h, w, c = 28, 28, 1
        # z noise size
        zn_size = 88

        # code latent variables distribution
        zc_distrib = O.distrib.MultinomialDistribution('cat', 10)
        zc_distrib *= O.distrib.GaussianDistributionWithUniformSample('code_a', 1, nr_num_samples=5)
        zc_distrib *= O.distrib.GaussianDistributionWithUniformSample('code_b', 1, nr_num_samples=5)
        net.zc_distrib = zc_distrib

        # prior: the assumption how the factors are presented in the dataset
        prior = O.constant([0.1] * 10 + [0, 0], dtype='float32', shape=[12], name='prior')
        batch_prior = O.tile(prior.add_axis(0), [g_batch_size, 1], name='batch_prior')

        net.zc_distrib_num_prior = np.array([0.1] * 10 + [0, 0], dtype='float32')

        dpc = env.create_dpcontroller()
        with dpc.activate():
            def inputs():
                img = O.placeholder('img', shape=(None, h, w, c))
                # only for demo-time
                zc = O.placeholder('zc', shape=(1, net.zc_distrib.sample_size))
                return [img, zc]

            def generator(z):
                w_init = tf.truncated_normal_initializer(stddev=0.02)
                with O.argscope(O.conv2d, O.deconv2d, kernel=4, stride=2, W=w_init),\
                     O.argscope(O.fc, W=w_init):

                    _ = z
                    _ = O.fc('fc1', _, 1024, nonlin=O.bn_relu)
                    _ = O.fc('fc2', _, 128 * 7 * 7, nonlin=O.bn_relu)
                    _ = O.reshape(_, [-1, 7, 7, 128])
                    _ = O.deconv2d('deconv1', _, 64, nonlin=O.bn_relu)
                    _ = O.deconv2d('deconv2', _, 1)
                    _ = O.sigmoid(_, 'out')
                return _

            def discriminator(img):
                w_init = tf.truncated_normal_initializer(stddev=0.02)
                with O.argscope(O.conv2d, O.deconv2d, kernel=4, stride=2, W=w_init),\
                     O.argscope(O.fc, W=w_init),\
                     O.argscope(O.leaky_relu, alpha=0.2):

                    _ = img
                    _ = O.conv2d('conv1', _, 64, nonlin=O.leaky_relu)
                    _ = O.conv2d('conv2', _, 128, nonlin=O.bn_nonlin)
                    _ = O.leaky_relu(_)
                    _ = O.fc('fc1', _, 1024, nonlin=O.bn_nonlin)
                    _ = O.leaky_relu(_)

                    with tf.variable_scope('score'):
                        logits = O.fc('fct', _, 1)

                    with tf.variable_scope('code'):
                        _ = O.fc('fc1', _, 128, nonlin=O.bn_nonlin)
                        _ = O.leaky_relu(_)
                        code = O.fc('fc2', _, zc_distrib.param_size)

                return logits, code

            def forward(x, zc):
                if env.phase is env.Phase.TRAIN:
                    zc = zc_distrib.sample(g_batch_size, prior)

                zn = O.random_normal([g_batch_size, zn_size], -1 , 1)
                z = O.concat([zc, zn], axis=1, name='z')
                
                with tf.variable_scope(GANGraphKeys.GENERATOR_VARIABLES):
                    x_given_z = generator(z)

                with tf.variable_scope(GANGraphKeys.DISCRIMINATOR_VARIABLES):
                    logits_fake, code_fake = discriminator(x_given_z)
                    score_fake = O.sigmoid(logits_fake)

                dpc.add_output(x_given_z, name='output')
                dpc.add_output(score_fake, name='score')
                dpc.add_output(code_fake, name='code')

                if env.phase is env.Phase.TRAIN:
                    with tf.variable_scope(GANGraphKeys.DISCRIMINATOR_VARIABLES, reuse=True):
                        logits_real, code_real = discriminator(x)
                        score_real = O.sigmoid(logits_real)

                    # build loss
                    with tf.variable_scope('loss'):
                        d_loss_real = O.sigmoid_cross_entropy_with_logits(
                            logits=logits_real, labels=O.ones_like(logits_real)).mean()
                        d_loss_fake = O.sigmoid_cross_entropy_with_logits(
                            logits=logits_fake, labels=O.zeros_like(logits_fake)).mean()
                        g_loss = O.sigmoid_cross_entropy_with_logits(
                            logits=logits_fake, labels=O.ones_like(logits_fake)).mean()

                        entropy = zc_distrib.cross_entropy(zc, batch_prior)
                        cond_entropy = zc_distrib.cross_entropy(zc, code_fake, process_theta=True)
                        info_gain = entropy - cond_entropy

                    d_acc_real = (score_real > 0.5).astype('float32').mean()
                    d_acc_fake = (score_fake < 0.5).astype('float32').mean()
                    g_accuracy = (score_fake > 0.5).astype('float32').mean()

                    d_accuracy = .5 * (d_acc_real + d_acc_fake)
                    d_loss = .5 * (d_loss_real + d_loss_fake)

                    d_loss -= info_gain
                    g_loss -= info_gain

                    dpc.add_output(d_loss, name='d_loss', reduce_method='sum')
                    dpc.add_output(d_accuracy, name='d_accuracy', reduce_method='sum')
                    dpc.add_output(d_acc_real, name='d_acc_real', reduce_method='sum')
                    dpc.add_output(d_acc_fake, name='d_acc_fake', reduce_method='sum')
                    dpc.add_output(g_loss, name='g_loss', reduce_method='sum')
                    dpc.add_output(g_accuracy, name='g_accuracy', reduce_method='sum')
                    dpc.add_output(info_gain, name='g_info_gain', reduce_method='sum')

            dpc.set_input_maker(inputs).set_forward_func(forward)

        if env.phase is env.Phase.TRAIN:
            for acc in ['d_accuracy', 'd_acc_real', 'd_acc_fake']:
                summary.scalar(acc, dpc.outputs[acc], collections=[GANGraphKeys.DISCRIMINATOR_SUMMARIES])
            summary.scalar('g_accuracy', dpc.outputs['g_accuracy'], collections=[GANGraphKeys.GENERATOR_SUMMARIES])
            summary.scalar('g_info_gain', dpc.outputs['g_info_gain'], collections=[GANGraphKeys.GENERATOR_SUMMARIES])

        net.add_all_dpc_outputs(dpc)


def make_optimizer(env):
    lr = optimizer.base.make_optimizer_variable('learning_rate', get_env('trainer.learning_rate'))

    wrapper = optimizer.OptimizerWrapper()
    # generator learns 5 times faster
    wrapper.set_base_optimizer(optimizer.base.AdamOptimizer(lr * 5., beta1=0.5, epsilon=1e-6))
    wrapper.append_grad_modifier(optimizer.grad_modifier.LearningRateMultiplier([
        ('*/b', 2.0),
    ]))
    env.set_g_optimizer(wrapper)
    wrapper = optimizer.OptimizerWrapper()
    wrapper.set_base_optimizer(optimizer.base.AdamOptimizer(lr, beta1=0.5, epsilon=1e-6))
    wrapper.append_grad_modifier(optimizer.grad_modifier.LearningRateMultiplier([
        ('*/b', 2.0),
    ]))
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

