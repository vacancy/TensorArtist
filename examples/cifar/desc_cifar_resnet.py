# -*- coding:utf8 -*-
# File   : desc_cifar.py
# Author : Jiayuan Mao
#          Honghua Dong
# Email  : maojiayuan@gmail.com
#          dhh19951@gmail,com
# Date   : 2/27/17
#
# This file is part of TensorArtist.

from tartist.core import get_env, get_logger
from tartist.core.utils.naming import get_dump_directory, get_data_directory
from tartist.nn import opr as O, optimizer, summary

import functools

logger = get_logger(__file__)

__envs__ = {
    'dir': {
        'root': get_dump_directory(__file__),
        'data': get_data_directory('WellKnown/cifar')
    },

    'dataset': {
        'nr_classes' : 10
    },

    'trainer': {
        'learning_rate': 0.01,

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
        n = 2
        nr_classes = get_env('dataset.nr_classes')

        conv2d = functools.partial(O.conv2d, kernel=3, use_bias=False, padding='SAME')
        conv_bn_relu = functools.partial(conv2d, nonlin=O.bn_relu)

        dpc = env.create_dpcontroller()
        with dpc.activate():
            def inputs():
                h, w, c = 32, 32, 3
                img = O.placeholder('img', shape=(None, h, w, c))
                return [img]

            def residual(name, x, first=False, inc_dim=False):
                in_channel = x.static_shape[3]
                out_channel = in_channel
                stride = 1
                if inc_dim:
                    out_channel = in_channel * 2
                    stride = 2
                with env.variable_scope(name):
                    _ = x if first else O.bn_relu(x)
                    _ = conv_bn_relu('conv1', _, out_channel, stride=stride)
                    _ = conv2d('conv2', _, out_channel)
                    if inc_dim:
                        x = O.pooling2d('pool', x, kernel=2)
                        x = O.pad(x, [[0, 0], [0, 0], [0, 0], [in_channel // 2, in_channel // 2]])
                print(name, x.static_shape)
                _ = _ + x
                return _

            def forward(img):
                _ = img / 128.0 - 1.0
                _ = conv_bn_relu('conv0', _, 16)
                _ = residual('res1.0', _, first=True)
                for i in range(1, n):
                    _ = residual('res1.{}'.format(i), _)
                _ = residual('res2.0', _, inc_dim=True)
                for i in range(1, n):
                    _ = residual('res2.{}'.format(i), _)
                _ = residual('res3.0', _, inc_dim=True)
                for i in range(1, n):
                    _ = residual('res3.{}'.format(i), _)

                _ = O.batch_norm('bn_last', _)
                _ = O.relu(_)

                _ = _.mean(axis=[1, 2]) # global avg pool

                dpc.add_output(_, name='feature')

            dpc.set_input_maker(inputs).set_forward_func(forward)

        _ = dpc.outputs['feature']
        _ = O.fc('linear', _, nr_classes)

        prob = O.softmax(_, name='prob')
        pred = _.argmax(axis=1).astype('int32', name='pred')
        net.add_output(prob)
        net.add_output(pred)

        if env.phase is env.Phase.TRAIN:
            label = O.placeholder('label', shape=(None, ), dtype='int32')
            loss = O.sparse_softmax_cross_entropy_with_logits(logits=_, labels=label).mean()
            loss = O.identity(loss, name='loss')
            net.set_loss(loss)

            accuracy = O.eq(label, pred).astype('float32').mean()
            error = 1. - accuracy

            summary.scalar('accuracy', accuracy)
            summary.scalar('error', error)
            summary.inference.scalar('loss', loss)
            summary.inference.scalar('accuracy', accuracy)
            summary.inference.scalar('error', error)


def make_optimizer(env):
    wrapper = optimizer.OptimizerWrapper()
    wrapper.set_base_optimizer(optimizer.base.MomentumOptimizer(get_env('trainer.learning_rate'), 0.9))
    wrapper.append_grad_modifier(optimizer.grad_modifier.LearningRateMultiplier([
        ('*/b', 2.0),
    ]))
    env.set_optimizer(wrapper)

from data_provider import *


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
