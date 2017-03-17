# -*- coding:utf8 -*-
# File   : desc_mnist.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/30/16
#
# This file is part of TensorArtist

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
        'learning_rate': 0.01,

        'batch_size': 100,
        'epoch_size': 500,
        'nr_epochs': 10,

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

        dpc = env.create_dpcontroller()
        with dpc.activate():
            def inputs():
                h, w, c = 28, 28, 1
                img = O.placeholder('img', shape=(None, h, w, c))
                return [img]

            def forward(img):
                _ = img
                _ = O.conv2d('conv1', _, 4, (3, 3), stride=2, padding='SAME', nonlin=O.relu)
                # shape = (14, 14)
                _ = O.conv2d('conv2', _, 8, (3, 3), stride=2, padding='SAME', nonlin=O.relu)
                # shape = (7, 7)
                _ = O.fc('fc', _, 392)
                _ = _.reshape([-1, 7, 7, 8])
                _ = O.deconv2d('deconv1', _, 4, (3, 3), stride=2, padding='SAME', nonlin=O.relu)
                # shape = (14, 14)
                _ = O.deconv2d('deconv2', _, 1, (3, 3), stride=2, padding='SAME', nonlin=O.sigmoid)
                # shape = (28, 28)
                out = _

                loss = O.raw_cross_entropy_prob('raw_loss', out, img)
                loss = O.get_pn_balanced_loss('loss', loss, img)
                dpc.add_output(out, name='output')
                dpc.add_output(loss, name='loss', reduce_method='sum')

            dpc.set_input_maker(inputs).set_forward_func(forward)

        net.add_all_dpc_outputs(dpc, loss_name='loss')

        if env.phase is env.Phase.TRAIN:
            summary.inference.scalar('loss', net.loss)


def make_optimizer(env):
    wrapper = optimizer.OptimizerWrapper()
    wrapper.set_base_optimizer(optimizer.base.AdamOptimizer(get_env('trainer.learning_rate'), beta1=0.9))
    wrapper.append_grad_modifier(optimizer.grad_modifier.LearningRateMultiplier([
        ('*/b', 2.0),
    ]))
    env.set_optimizer(wrapper)


from data_provider_mnist import *


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

