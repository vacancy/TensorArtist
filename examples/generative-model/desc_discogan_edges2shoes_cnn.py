# -*- coding:utf8 -*-
# File   : desc_discogan_edges2shoes_cnn.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 4/2/17
#
# This file is part of TensorArtist.

import re

from tartist.app import gan
from tartist.app.gan import GANGraphKeys
from tartist.core import get_env, get_logger
from tartist.core.utils.naming import get_dump_directory, get_data_directory
from tartist.nn import opr as O, optimizer, summary

logger = get_logger(__file__)

__envs__ = {
    'dir': {
        'root': get_dump_directory(__file__),
        'data': get_data_directory('Pix2Pix/edges2shoes')
    },
    'dataset': {
        'name': 'edges2shoes',
        'db_a': 'train_edges_db',
        'db_b': 'train_shoes_db',
    },
    'trainer': {
        'learning_rate': 2e-4,

        'batch_size': 32,
        'epoch_size': 100,
        'nr_epochs': 200,
        'nr_g_per_iter': 2,

        'env_flags': {
            'log_device_placement': False
        }
    }
}

__trainer_cls__ = gan.GANTrainer
__trainer_env_cls__ = gan.GANTrainerEnv


def make_network(env):
    with env.create_network() as net:
        h, w, c = 64, 64, 3

        def bn_leaky_relu(x, name='bn_leaky_relu'):
            with env.name_scope(name):
                return O.leaky_relu(O.bn_nonlin(x))

        dpc = env.create_dpcontroller()
        with dpc.activate():
            def inputs():
                img_a = O.placeholder('img_a', shape=(None, h, w, c))
                img_b = O.placeholder('img_b', shape=(None, h, w, c))
                return [img_a, img_b]

            def encoder(x):
                w_init = O.truncated_normal_initializer(stddev=0.02)
                with O.argscope(O.conv2d, O.deconv2d, kernel=4, stride=2, W=w_init),\
                     O.argscope(O.fc, W=w_init),\
                     O.argscope(O.leaky_relu, alpha=0.2):

                    _ = x
                    _ = O.conv2d('conv1', _, 64, nonlin=O.leaky_relu)
                    _ = O.conv2d('conv2', _, 128, nonlin=bn_leaky_relu, use_bias=False)
                    _ = O.conv2d('conv3', _, 256, nonlin=bn_leaky_relu, use_bias=False)
                    _ = O.conv2d('conv4', _, 512, nonlin=bn_leaky_relu, use_bias=False)
                    z = _
                return z

            def decoder(z):
                w_init = O.truncated_normal_initializer(stddev=0.02)
                with O.argscope(O.conv2d, O.deconv2d, kernel=4, stride=2, W=w_init),\
                     O.argscope(O.fc, W=w_init):

                    _ = z
                    _ = O.deconv2d('deconv1', _, 256, nonlin=O.bn_relu)
                    _ = O.deconv2d('deconv2', _, 128, nonlin=O.bn_relu)
                    _ = O.deconv2d('deconv3', _, 64, nonlin=O.bn_relu)
                    _ = O.deconv2d('deconv4', _, c)
                    _ = O.sigmoid(_, name='out')
                x = _
                return x

            def generator(x, name, reuse):
                with env.variable_scope(GANGraphKeys.GENERATOR_VARIABLES, reuse=reuse):
                    with env.variable_scope(name):
                        z = encoder(x)
                        y = decoder(z)
                return y

            def discriminator(x, name, reuse):
                with env.variable_scope(GANGraphKeys.DISCRIMINATOR_VARIABLES, reuse=reuse):
                    with env.variable_scope(name):
                        z = encoder(x)
                        logit = O.fc('fc', z, 1)
                return logit

            def forward(img_a, img_b):
                img_a /= 255.
                img_b /= 255.

                img_ab = generator(img_a, name='atob', reuse=False)
                img_ba = generator(img_b, name='btoa', reuse=False)
                img_aba = generator(img_ab, name='btoa', reuse=True)
                img_bab = generator(img_ba, name='atob', reuse=True)

                logit_fake_a = discriminator(img_ba, name='a', reuse=False)
                logit_fake_b = discriminator(img_ab, name='b', reuse=False)

                score_fake_a = O.sigmoid(logit_fake_a)
                score_fake_b = O.sigmoid(logit_fake_b)

                for name in ['img_ab', 'img_ba', 'img_aba', 'img_bab', 'score_fake_a', 'score_fake_b']:
                    dpc.add_output(locals()[name], name=name)

                if env.phase is env.Phase.TRAIN:
                    logit_real_a = discriminator(img_a, name='a', reuse=True)
                    logit_real_b = discriminator(img_b, name='b', reuse=True)
                    score_real_a = O.sigmoid(logit_real_a)
                    score_real_b = O.sigmoid(logit_real_b)

                    all_g_loss = 0.
                    all_d_loss = 0.
                    r_loss_ratio = 0.9

                    for pair_name, (real, fake), (logit_real, logit_fake), (score_real, score_fake) in zip(
                            ['lossa', 'lossb'],
                            [(img_a, img_aba), (img_b, img_bab)],
                            [(logit_real_a, logit_fake_a), (logit_real_b, logit_fake_b)],
                            [(score_real_a, score_fake_a), (score_real_b, score_fake_b)]):

                        with env.name_scope(pair_name):
                            d_loss_real = O.sigmoid_cross_entropy_with_logits(logits=logit_real, labels=O.ones_like(logit_real)).mean(name='d_loss_real')
                            d_loss_fake = O.sigmoid_cross_entropy_with_logits(logits=logit_fake, labels=O.zeros_like(logit_fake)).mean(name='d_loss_fake')
                            g_loss = O.sigmoid_cross_entropy_with_logits(logits=logit_fake, labels=O.ones_like(logit_fake)).mean(name='g_loss')

                            d_acc_real = (score_real > 0.5).astype('float32').mean(name='d_acc_real')
                            d_acc_fake = (score_fake < 0.5).astype('float32').mean(name='d_acc_fake')
                            g_accuracy = (score_fake > 0.5).astype('float32').mean(name='g_accuracy')

                            d_accuracy = O.identity(.5 * (d_acc_real + d_acc_fake), name='d_accuracy')
                            d_loss = O.identity(.5 * (d_loss_real + d_loss_fake), name='d_loss')

                            # r_loss = O.raw_l2_loss('raw_r_loss', real, fake).flatten2().sum(axis=1).mean(name='r_loss')
                            r_loss = O.raw_l2_loss('raw_r_loss', real, fake).mean(name='r_loss')

                            # all_g_loss += g_loss + r_loss
                            all_g_loss += (1 - r_loss_ratio) * g_loss + r_loss_ratio * r_loss
                            all_d_loss += d_loss

                        for v in [d_loss_real, d_loss_fake, g_loss, d_acc_real, d_acc_fake, g_accuracy, d_accuracy, d_loss, r_loss]:
                            dpc.add_output(v, name=re.sub('^tower/\d+/', '', v.name)[:-2], reduce_method='sum')

                    dpc.add_output(all_g_loss, name='g_loss', reduce_method='sum')
                    dpc.add_output(all_d_loss, name='d_loss', reduce_method='sum')

            dpc.set_input_maker(inputs).set_forward_func(forward)

        if env.phase is env.Phase.TRAIN:
            for p in ['lossa', 'lossb']:
                for v in ['d_loss_real', 'd_loss_fake', 'd_acc_real', 'd_acc_fake', 'd_accuracy', 'd_loss']:
                    name = p + '/' + v
                    summary.scalar(name, dpc.outputs[name], collections=[GANGraphKeys.DISCRIMINATOR_SUMMARIES])
                for v in ['g_loss', 'g_accuracy', 'r_loss']:
                    name = p + '/' + v
                    summary.scalar(name, dpc.outputs[name], collections=[GANGraphKeys.GENERATOR_SUMMARIES])

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


from data_provider_discogan import *


def main_train(trainer):
    from tartist.plugins.trainer_enhancer import summary
    summary.enable_summary_history(trainer)
    summary.enable_echo_summary_scalar(trainer)

    from tartist.plugins.trainer_enhancer import progress
    progress.enable_epoch_progress(trainer)

    from tartist.plugins.trainer_enhancer import snapshot
    snapshot.enable_snapshot_saver(trainer)

    trainer.train()
