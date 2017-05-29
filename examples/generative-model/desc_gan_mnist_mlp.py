# -*- coding:utf8 -*-
# File   : desc_gan_mnist_mlp.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/17/17
# 
# This file is part of TensorArtist

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

__trainer_cls__ = gan.GANTrainer
__trainer_env_cls__ = gan.GANTrainerEnv


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
                with env.variable_scope(GANGraphKeys.GENERATOR_VARIABLES):
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
                    with env.variable_scope(GANGraphKeys.DISCRIMINATOR_VARIABLES):
                        logits_real = discriminator(img).flatten()
                        score_real = O.sigmoid(logits_real)

                with env.variable_scope(GANGraphKeys.DISCRIMINATOR_VARIABLES, reuse=is_train):
                    logits_fake = discriminator(x_given_z).flatten()
                    score_fake = O.sigmoid(logits_fake)

                if is_train:
                    # build loss
                    with env.variable_scope('loss'):
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

                dpc.add_output(x_given_z, name='output')
                dpc.add_output(score_fake, name='score')

            dpc.set_input_maker(inputs).set_forward_func(forward)

        if is_train:
            for acc in ['d_accuracy', 'd_acc_real', 'd_acc_fake']:
                summary.scalar(acc, dpc.outputs[acc], collections=[GANGraphKeys.DISCRIMINATOR_SUMMARIES])
            summary.scalar('g_accuracy', dpc.outputs['g_accuracy'], collections=[GANGraphKeys.GENERATOR_SUMMARIES])

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

