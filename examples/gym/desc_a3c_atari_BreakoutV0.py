# -*- coding:utf8 -*-
# File   : desc_a3c_atari_BreakoutV0.py
# Author : Jiayuan Mao
#          Honghua Dong
# Email  : maojiayuan@gmail.com
#          dhh19951@gmail.com
# Date   : 3/18/17
#
# This file is part of TensorArtist.

"""This reproduction of A3C is based on ppwwyyxx's reproduction in his TensorPack framework.
Credit to : https://github.com/ppwwyyxx/tensorpack/tree/master/examples/A3C-Gym"""

import collections
import functools
import os
import queue

import numpy as np

from tartist import random, image
from tartist.app import rl
from tartist.core import get_env, get_logger
from tartist.core.utils.cache import cached_result
from tartist.core.utils.naming import get_dump_directory
from tartist.data import flow
from tartist.nn import opr as O, optimizer, summary

logger = get_logger(__file__)

__envs__ = {
    'dir': {
        'root': get_dump_directory(__file__),
    },

    'a3c': {
        'env_name': 'Breakout-v0',

        'input_shape': (84, 84),
        'frame_history': 4,
        'limit_length': 40000,

        # gamma and acc_step in future_reward
        'gamma': 0.99,
        'acc_step': 5,

        # async training data collector
        'nr_players': 50,
        'nr_predictors': 2,
        'predictor': {
            'batch_size': 16,
            'outputs_name': ['value', 'policy_explore']
        },

        'inference': {
            'nr_plays': 20,
            'max_stuck_repeat': 30
        },
        'demo': {
            'nr_plays': 5
        }
    },

    'trainer': {
        'learning_rate': 0.001,

        'batch_size': 128,
        'epoch_size': 1000,
        'nr_epochs': 200,
    },
    'demo': {
        'customized': True
    }
}

__trainer_cls__ = rl.train.A3CTrainer
__trainer_env_cls__ = rl.train.A3CTrainerEnv


def make_network(env):
    is_train = env.phase is env.Phase.TRAIN
    if is_train:
        slave_devices = env.slave_devices
        env.set_slave_devices([])

    with env.create_network() as net:
        h, w, c = get_input_shape()

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
        policy = O.fc('fc_policy', _, get_player_nr_actions())
        value = O.fc('fc_value', _, 1)

        expf = O.scalar('explore_factor', 1, trainable=False)
        policy_explore = O.softmax(policy * expf, name='policy_explore')

        policy = O.softmax(policy, name='policy')
        value = value.remove_axis(1, name='value')

        net.add_output(policy_explore, name='policy_explore')
        net.add_output(policy, name='policy')
        net.add_output(value, name='value')

        if env.phase is env.Phase.TRAIN:
            action = O.placeholder('action', shape=(None, ), dtype='int64')
            future_reward = O.placeholder('future_reward', shape=(None, ))

            log_policy = O.log(policy + 1e-6)
            log_pi_a_given_s = (log_policy * O.one_hot(action, get_player_nr_actions())).sum(axis=1)
            advantage = (future_reward - O.zero_grad(value)).rename('advantage')
            policy_cost = (log_pi_a_given_s * advantage).mean(name='policy_cost')
            xentropy_cost = (-policy * log_policy).sum(axis=1).mean(name='xentropy_cost')
            value_loss = O.raw_l2_loss('raw_value_loss', future_reward, value).mean(name='value_loss')
            entropy_beta = O.scalar('entropy_beta', 0.01, trainable=False)
            loss = O.add_n([-policy_cost, -xentropy_cost * entropy_beta, value_loss], name='loss')

            net.set_loss(loss)

            for v in [policy_cost, xentropy_cost, value_loss,
                      value.mean(name='predict_value'), advantage.rms(name='rms_advantage'), loss]:
                summary.scalar(v)

    if is_train:
        env.set_slave_devices(slave_devices)


def make_player(is_train=True, dump_dir=None):
    def resize_state(s):
        return image.resize(s, get_env('a3c.input_shape'), interpolation='NEAREST')

    p = rl.GymRLEnviron(get_env('a3c.env_name'), dump_dir=dump_dir)
    p = rl.MapStateProxyRLEnviron(p, resize_state)
    p = rl.GymHistoryProxyRLEnviron(p, get_env('a3c.frame_history'))

    p = rl.LimitLengthProxyRLEnviron(p, get_env('a3c.limit_length'))
    if is_train:
        p = rl.AutoRestartProxyRLEnviron(p)
    else:
        p = rl.GymPreventStuckProxyRLEnviron(p, get_env('a3c.inference.max_stuck_repeat'), 1)
    return p


def make_optimizer(env):
    lr = optimizer.base.make_optimizer_variable('learning_rate', get_env('trainer.learning_rate'))

    wrapper = optimizer.OptimizerWrapper()
    wrapper.set_base_optimizer(optimizer.base.AdamOptimizer(lr, epsilon=1e-3))
    wrapper.append_grad_modifier(optimizer.grad_modifier.LearningRateMultiplier([
        ('*/b', 2.0),
    ]))
    wrapper.append_grad_modifier(optimizer.grad_modifier.GlobalGradClipByAvgNorm(0.1))
    env.set_optimizer(wrapper)


def make_dataflow_train(env):
    batch_size = get_env('trainer.batch_size')

    df = flow.QueueDataFlow(env.data_queue)
    df = flow.BatchDataFlow(df, batch_size, sample_dict={
        'state': np.empty((batch_size, ) + get_input_shape(), dtype='float32'),
        'action': np.empty((batch_size, ), dtype='int32'),
        'future_reward': np.empty((batch_size, ), dtype='float32')
    })
    return df


@cached_result
def get_player_nr_actions():
    p = make_player()
    n = p.action_space.nr_actions
    del p
    return n


@cached_result
def get_input_shape():
    input_shape = get_env('a3c.input_shape')
    frame_history = get_env('a3c.frame_history')
    h, w, c = input_shape[0], input_shape[1], 3 * frame_history
    return h, w, c


def player_func(pid, requester):
    player = make_player()
    player.restart()
    state = player.current_state
    reward = 0
    is_over = False
    with requester.activate():
        while True:
            action = requester.query('data', (state, reward, is_over))
            reward, is_over = player.action(action)

            if len(player.stats['score']) > 0:
                score = player.stats['score'][-1]
                requester.query('stat', {'async/score': score}, do_recv=False)
                player.clear_stats()
            state = player.current_state


def inference_player_func(pid, requester):
    player = make_player(is_train=False)

    def get_action(o):
        action = requester.query('data', (o, ))
        return action

    with requester.activate():
        player.play_one_episode(get_action)
        score = player.stats['score'][-1]
        requester.query('stat', {'async/inference/score': score}, do_recv=False)


def _predictor_func(pid, router, task_queue, func, is_inference=False):
    batch_size = get_env('a3c.predictor.batch_size')
    batched_state = np.empty((batch_size, ) + get_input_shape(), dtype='float32')

    while True:
        callbacks = []
        nr_total = 0
        for i in range(batch_size):
            if i == 0 or not is_inference:
                identifier, inp, callback = task_queue.get()
            else:
                try:
                    identifier, inp, callback = task_queue.get_nowait()
                except queue.Empty:
                    break

            batched_state[i] = inp[0]
            callbacks.append(callback)
            nr_total += 1

        out = func(state=batched_state)
        for i in range(nr_total):
            policy = out['policy_explore'][i]
            if is_inference:
                # during inference, policy should be out['policy'][i]
                # but these two are equivalent under argmax operation
                # and we can only compile 'policy_explore' in output
                action = policy.argmax()
            else:
                action = random.choice(len(policy), p=policy)

            callbacks[i](action, out['value'][i])


def make_a3c_configs(env):
    from common_a3c import on_data_func, on_stat_func
    predictor_func = functools.partial(_predictor_func, is_inference=False)

    env.player_master.player_func = player_func
    env.player_master.predictor_func = predictor_func
    env.player_master.on_data_func = on_data_func
    env.player_master.on_stat_func = on_stat_func

    # currently we don't use multi-proc inference, so these settings are not used at all
    if False:
        from common_a3c import inference_on_data_func, inference_on_stat_func
        inference_predictor_func = functools.partial(_predictor_func, is_inference=True)

        env.inference_player_master.player_func = inference_player_func
        env.inference_player_master.predictor_func = inference_predictor_func
        env.inference_player_master.on_data_func = inference_on_data_func
        env.inference_player_master.on_stat_func = inference_on_stat_func

    env.players_history = collections.defaultdict(list)


def main_train(trainer):
    from tartist.plugins.trainer_enhancer import summary
    summary.enable_summary_history(trainer, extra_summary_types={
        'async/score': 'async_scalar',
        'async/inference/score': 'async_scalar'
    })
    summary.enable_echo_summary_scalar(trainer, summary_spec={
        'async/score': ['avg', 'max'],
        'async/inference/score': ['avg', 'max']
    })

    from tartist.plugins.trainer_enhancer import progress
    progress.enable_epoch_progress(trainer)

    from tartist.plugins.trainer_enhancer import snapshot
    snapshot.enable_snapshot_saver(trainer)

    from tartist.core import register_event
    from common_a3c import main_inference_play_multithread

    def on_epoch_after(trainer):
        if trainer.epoch > 0 and trainer.epoch % 2 == 0:
            main_inference_play_multithread(trainer, make_player=make_player)

    # this one should run before monitor
    register_event(trainer, 'epoch:after', on_epoch_after, priority=5)

    trainer.train()


def main_demo(env, func):
    dump_dir = get_env('dir.demo', os.path.join(get_env('dir.root'), 'demo'))
    logger.info('demo dump dir: {}'.format(dump_dir))
    player = make_player(is_train=False, dump_dir=dump_dir)
    repeat_time = get_env('a3c.demo.nr_plays', 1)

    def get_action(inp, func=func):
        action = func(**{'state':[[inp]]})['policy'][0].argmax()
        return action

    for i in range(repeat_time):
        player.play_one_episode(get_action)
        logger.info('#{} play score={}'.format(i, player.stats['score'][-1]))
