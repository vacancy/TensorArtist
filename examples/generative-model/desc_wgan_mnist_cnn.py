# -*- coding:utf8 -*-
# File   : desc_wgan_mnist_cnn.py
# Author : Jiayuan Mao
#          Honghua Dong
# Email  : maojiayuan@gmail.com
#          dhh19951@gmail.com
# Date   : 4/1/17
#
# This file is part of TensorArtist.

from tartist.app import gan
from tartist.app.gan import GANGraphKeys
from tartist.core import get_env, get_logger
from tartist.core.utils.naming import get_dump_directory, get_data_directory
from tartist.nn import opr as O, optimizer

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
        'learning_rate': 1e-4,

        'batch_size': 64,
        'epoch_size': 500,
        'nr_epochs': 200,

        'env_flags': {
            'log_device_placement': False
        }
    }
}

__trainer_cls__ = gan.GANTrainer
__trainer_env_cls__ = gan.GANTrainerEnv


def make_network(env):
    with env.create_network() as net:
        h, w, c = 28, 28, 1
        z_dim = 100

        dpc = env.create_dpcontroller()
        with dpc.activate():
            def inputs():
                img = O.placeholder('img', shape=(None, h, w, c))
                if env.phase is env.Phase.TRAIN:
                    return [img]
                else:
                    return []

            def generator(z):
                w_init = O.truncated_normal_initializer(stddev=0.02)
                with O.argscope(O.conv2d, O.deconv2d, kernel=4, stride=2, W=w_init),\
                     O.argscope(O.fc, W=w_init):

                    _ = z
                    _ = O.fc('fc1', _, 1024, nonlin=O.bn_relu)
                    _ = O.fc('fc2', _, 128 * 7 * 7, nonlin=O.bn_relu)
                    _ = O.reshape(_, [-1, 7, 7, 128])
                    _ = O.deconv2d('deconv1', _, 64, nonlin=O.bn_relu)
                    _ = O.deconv2d('deconv2', _, 1)
                    _ = O.sigmoid(_, name='out')
                return _

            def discriminator(img):
                w_init = O.truncated_normal_initializer(stddev=0.02)
                with O.argscope(O.conv2d, O.deconv2d, kernel=4, stride=2, W=w_init),\
                     O.argscope(O.fc, W=w_init),\
                     O.argscope(O.leaky_relu, alpha=0.2):

                    _ = img
                    _ = O.conv2d('conv1', _, 64, nonlin=O.leaky_relu)
                    _ = O.conv2d('conv2', _, 128, nonlin=O.bn_nonlin)
                    _ = O.leaky_relu(_)
                    _ = O.fc('fc1', _, 1024, nonlin=O.bn_nonlin)
                    _ = O.leaky_relu(_)
                    _ = O.fc('fct', _, 1)
                return _

            def forward(x=None):
                g_batch_size = get_env('trainer.batch_size') if env.phase is env.Phase.TRAIN else 1
                z = O.random_normal([g_batch_size, z_dim])

                with env.variable_scope(GANGraphKeys.GENERATOR_VARIABLES):
                    img_gen = generator(z)
                # tf.summary.image('generated-samples', img_gen, max_outputs=30)

                with env.variable_scope(GANGraphKeys.DISCRIMINATOR_VARIABLES):
                    logits_fake = discriminator(img_gen)
                dpc.add_output(img_gen, name='output')

                if env.phase is env.Phase.TRAIN:
                    with env.variable_scope(GANGraphKeys.DISCRIMINATOR_VARIABLES, reuse=True):
                        logits_real = discriminator(x)

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
    # wrapper.append_grad_modifier(optimizer.grad_modifier.LearningRateMultiplier([
    #     ('*/b', 2.0),
    # ]))
    env.set_g_optimizer(wrapper)
    env.set_d_optimizer(wrapper)


def enable_param_clippping(trainer):
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


from data_provider_gan_mnist import *


def main_train(trainer):
    from tartist.plugins.trainer_enhancer import summary
    summary.enable_summary_history(trainer)
    summary.enable_echo_summary_scalar(trainer)

    from tartist.plugins.trainer_enhancer import progress
    progress.enable_epoch_progress(trainer)

    from tartist.plugins.trainer_enhancer import snapshot
    snapshot.enable_snapshot_saver(trainer)

    enable_param_clippping(trainer)

    trainer.train()
