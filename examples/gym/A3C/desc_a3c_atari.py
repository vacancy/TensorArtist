# -*- coding:utf8 -*-
# File   : desc_a3c_atari.py
# Author : Jiayuan Mao
#          Honghua Dong
# Email  : maojiayuan@gmail.com
#          dhh19951@gmail,com
# Date   : 3/18/17
#
# This file is part of TensorArtist

import numpy as np
import tensorflow as tf

from tartist.core import get_env, get_logger
from tartist.core.utils.naming import get_dump_directory, get_data_directory
from tartist.data import flow
from tartist.nn import opr as O, optimizer, summary, train
from tartist.nn.train.gan import GANGraphKeys


logger = get_logger(__file__)

__envs__ = {
    'dir': {
        'root': get_dump_directory(__file__),
        'data': get_data_directory('WellKnown/mnist')
    },

    'dataset': {
        'env_name': 'Breakout-v0',
        'input_shape': 84,
        'frame_history': 4,
    },

    'trainer': {
        'learning_rate': 2e-4,

        'batch_size': 128,
        'nr_epochs': 200,

        'gamma': 0.99,

        'env_flags': {
            'log_device_placement': False
        }
    }
}


def make_network(env):
    with env.create_network() as net:
        input_shape = get_env('dataset.input_shape')
        h, w, c = input_shape, input_shape, 12

        dpc = env.create_dpcontroller()
        with dpc.activate():
            def inputs():
                state = O.placeholder('state', shape=(None, h, w, c))
                return [state]

            def forward(x):
                _ = x / 255.0
                with O.argscope(O.conv2d, nonlin=O.relu):
                    _ = O.conv2d('conv0', _, 32, 5)
                    _ = O.max_pooling2d('pool0', _, 2)
                    _ = O.conv2d('conv1', _, 32, 5)
                    _ = O.max_pooling2d('pool1', _, 2)
                    _ = O.conv2d('conv2', _, 64, 4)
                    _ = O.max_pooling2d('pool2', _, 2)
                    _ = O.conv2d('conv3', _, 64, 3)

                dpc.add_output(_, name='feature')

            dpc.set_input_maker(inputs).set_forward_func(forward)

        _ = dpc.outputs['feature']
        _ = O.fc('fc0', _, 512, nonlin=O.p_relu)
        policy = O.fc('fc_policy', _, NUM_ACTIONS)
        value = O.fc('fc_value', _, 1)

        logits = O.softmax(policy, name='logits')
        value = O.squeeze(value, [1], name='value')

        expf = O.scalar('explore_factor', 1, trainable=False)
        logits_t = O.softmax(policy * expf, name='logits_t')

        net.add_output(logits_t)
        net.add_output(value)

        if env.phase is env.Phase.TRAIN:
            action = O.placeholder('action', shape=(None, ), dtype=tf.int32)
            future_reward = O.placeholder('future_reward', shape=(None, ))
            log_logits = O.log(logits + 1e-6)
            log_pi_a_given_s = (log_logits * tf.one_hot(action, NUM_ACTIONS)).sum(axis=1)
            advantage = future_reward - O.stop_gradient(value, name='advantage')
            policy_cost = (log_pi_a_given_s * advantage).mean(name='policy_cost')
            xentropy_cost = (-logits * log_logits).mean(name='xentropy_cost')
            value_loss = O.raw_l2_loss(future_reward, value).mean(name='value_loss')
            # value_loss = O.truediv(value_loss, future_reward.shape[0].astype('float32'), name='value_loss')
            entropy_beta = O.scalar('entropy_beta', 0.01, trainable=False)
            cost = tf.add_n([-policy_cost, -xentropy_cost * entropy_beta, value_loss], name='loss')

            for v in [policy_cost, xentropy_cost, value_loss, 
                    value.mean(name='predict_value'), advantage.rms(name='rms_advantage'), cost]:
                summary.scalar(v)


def player_func(i, requester):
    player = make_player()
    state = player.current_state
    while True:
        # must have an action field
        response = requester.query({
            'action': 'data',
            'state': state
        })
        action = response['action']
        state, reward = player.action(action)


def predictor_func(i, queue, func):
    while True:
        inp, cb = queue.get()
        out = func(inp)
        cb(out)


def on_data_func(env, player_router, identifier, inp_data):
    predictor_queue = env.predictors_queue
    data_queue = env.data_queue

    def callback(out_data):
        # data_queue.put_nowait(some_composed_data)
        # player_router.send(out_data)


def make_a3c_configs(env):
    env.player_func = player_func
    env.predictor_func = predictor_func
    env.on_data_func = on_data_func


def make_optimizer(env):
    lr = optimizer.base.make_optimizer_variable('learning_rate', get_env('trainer.learning_rate'))

    wrapper = optimizer.OptimizerWrapper()
    # generator learns 5 times faster
    wrapper.set_base_optimizer(optimizer.base.AdamOptimizer(lr * 5., beta1=0.5, epsilon=1e-6))
    wrapper.append_grad_modifier(optimizer.grad_modifier.LearningRateMultiplier([
        ('*/b', 2.0),
    ]))
    env.set_optimizer(wrapper)


def make_dataflow_train(env):
    batch_size = get_env('trainer.batch_size')

    df = flow.QueueDataFlow(env.data_queue)
    df = flow.BatchDataFlow(df, batch_size, sample_dict={
        # write input here
    })
    return df


def main_train(trainer):
    from tartist.plugins.trainer_enhancer import summary
    summary.enable_summary_history(trainer)
    summary.enable_echo_summary_scalar(trainer)

    from tartist.plugins.trainer_enhancer import progress
    progress.enable_epoch_progress(trainer)

    from tartist.plugins.trainer_enhancer import snapshot
    snapshot.enable_snapshot_saver(trainer)

    trainer.train()

