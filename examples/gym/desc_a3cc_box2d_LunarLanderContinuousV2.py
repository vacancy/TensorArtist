# -*- coding:utf8 -*-
# File   : desc_a3cc_box2d_LunarLanderContinuousV2.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 5/31/17
#
# This file is part of TensorArtist.

"""
A3C-Continuous reproduction on Lunar Lander game. (OpenAI.Gym.Box2D.LunarLander)
This model does not follows the original settings in DeepMind's paper, which use:
    1. LSTM model.
    2. Episode-as-a-batch update.
    3. Gaussian distribution.
In this model, we included several tricks for the training:
    1. Truncated Laplacian distribution for policy.
    2. Positive advantage only update.
Details can be found in the code.
"""


import collections
import functools
import os
import queue

import numpy as np

from tartist import random
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
        'env_name': 'LunarLanderContinuous-v2',

        'nr_history_frames': 4,
        'max_nr_steps': None, # no limit length

        # Action space used for exploration strategy sampling
        # Instead of sampling from a truncated Laplacian distribution, we perform a simplified version via
        # discretizing the action space.
        'actor_space': np.array([
            np.linspace(-1, 1, 11),
            np.linspace(-1, 1, 11)
        ], dtype='float32'),

        # gamma and TD steps in future_reward
        'gamma': 0.99,
        'nr_td_steps': 5,

        # async training data collector
        'nr_players': 50,
        'nr_predictors': 2,
        'predictor': {
            'batch_size': 16,
            'outputs_name': ['value', 'policy_explore', 'policy']
        },

        'inference': {
            'nr_plays': 20,
        },
        'demo': {
            'nr_plays': 5
        }
    },
    'trainer': {
        'learning_rate': 0.001,

        'batch_size': 128,
        'epoch_size': 200,
        'nr_epochs': 100,
    }
}

__trainer_cls__ = rl.train.A3CTrainer
__trainer_env_cls__ = rl.train.A3CTrainerEnv


# normal pdf, not used (instead, use Laplace distribution)
def normal_pdf(x, mu, var):
    exponent = ((x - mu) ** 2.) / (var + 1e-4)
    prob = (1. / (2. * np.pi * var)) * O.exp(-exponent)
    return prob


def make_network(env):
    is_train = env.phase is env.Phase.TRAIN

    # device control: always use master device only for training session
    if is_train:
        slave_devices = env.slave_devices
        env.set_slave_devices([])
    
    with env.create_network() as net:
        input_length, = get_input_shape()
        action_length, = get_action_shape()

        dpc = env.create_dpcontroller()
        with dpc.activate():
            def inputs():
                state = O.placeholder('state', shape=(None, input_length))
                return [state]

            # forward policy network and value network separately (actor-critic)
            def forward(x):
                _ = x
                _ = O.fc('fcp1', _, 512, nonlin=O.relu)
                _ = O.fc('fcp2', _, 256, nonlin=O.relu)
                dpc.add_output(_, name='feature_p')

                _ = x
                _ = O.fc('fcv1', _, 512, nonlin=O.relu)
                _ = O.fc('fcv2', _, 256, nonlin=O.relu)
                dpc.add_output(_, name='feature_v')

            dpc.set_input_maker(inputs).set_forward_func(forward)

        _ = dpc.outputs['feature_p']
        # mu and std, assuming spherical covariance
        policy_mu = O.fc('fc_policy_mu', _, action_length)

        # In this example, we do not use variance. instead, we use fixed value.
        # policy_var = O.fc('fc_policy_var', _, 1, nonlin=O.softplus)
        # policy_var = O.tile(policy_var, [1, action_length], name='policy_var')
        # policy_std = O.sqrt(policy_var, name='policy_std')

        actor_space = get_env('a3c.actor_space')
        nr_bins = actor_space.shape[1]

        # Instead of using normal distribution, we use Laplacian distribution for policy.
        # And also, we are sampling from a truncated Laplacian distribution (only care the value in the
        # action space). To simplify the computation, we discretize the action space.
        actor_space = O.constant(actor_space)
        actor_space = O.tile(actor_space.add_axis(0), [policy_mu.shape[0], 1, 1])
        policy_mu3 = O.tile(policy_mu.add_axis(2), [1, 1, nr_bins])

        # policy_std3 = O.tile(policy_std.add_axis(2), [1, 1, nr_bins])
        # logits = O.abs(actor_space - policy_mu3) / (policy_std3 + 1e-2)

        # Here, we force the std of the policy to be 1.
        logits_explore = -O.abs(actor_space - policy_mu3)
        policy_explore = O.softmax(logits_explore)

        # Clip the policy for output
        action_range = get_action_range()
        action_range = tuple(map(O.constant, action_range))
        action_range = tuple(map(lambda x: O.tile(x.add_axis(0), [policy_mu.shape[0], 1]), action_range))
        policy_output = O.clip_by_value(policy_mu, *action_range)

        _ = dpc.outputs['feature_v']
        value = O.fc('fc_value', _, 1)
        value = value.remove_axis(1, name='value')

        # Note that, here the policy_explore is a discrete policy,
        # and policy is actually the continuous one.
        net.add_output(policy_explore, name='policy_explore')
        net.add_output(policy_output, name='policy')
        net.add_output(value, name='value')

        if is_train:
            action = O.placeholder('action', shape=(None, action_length), dtype='int64')
            future_reward = O.placeholder('future_reward', shape=(None, ))

            # Since we discretized the action space, use cross entropy here.
            log_policy = O.log(policy_explore + 1e-4)
            log_pi_a_given_s = (log_policy * O.one_hot(action, nr_bins)).sum(axis=2).sum(axis=1)
            advantage = (future_reward - O.zero_grad(value)).rename('advantage')

            # Important trick: using only positive advantage to perform gradient assent. This stabilizes the training.
            advantage = advantage * O.zero_grad((advantage > 0.).astype('float32'))
            policy_cost = (log_pi_a_given_s * advantage).mean(name='policy_cost')

            # As mentioned, there is no trainable variance.
            # xentropy_cost = (policy_std ** 2.).sum(axis=1).mean(name='xentropy_cost')
            # entropy_beta = O.scalar('entropy_beta', 0.1, trainable=False)

            value_loss = O.raw_smooth_l1_loss('raw_value_loss', future_reward, value).mean(name='value_loss')

            loss = O.add_n([-policy_cost, value_loss], name='loss')

            net.set_loss(loss)

            for v in [policy_cost, value_loss,
                      value.mean(name='predict_value'), advantage.rms(name='rms_advantage'), loss]:
                summary.scalar(v)

    if is_train:
        env.set_slave_devices(slave_devices)


def make_player(is_train=True, dump_dir=None):
    p = rl.GymRLEnviron(get_env('a3c.env_name'), dump_dir=dump_dir)
    p = rl.HistoryFrameProxyRLEnviron(p, get_env('a3c.nr_history_frames'))
    p = rl.LimitLengthProxyRLEnviron(p, get_env('a3c.max_nr_steps'))
    if is_train:
        p = rl.AutoRestartProxyRLEnviron(p)
    return p


def make_optimizer(env):
    lr = optimizer.base.make_optimizer_variable('learning_rate', get_env('trainer.learning_rate'))

    wrapper = optimizer.OptimizerWrapper()
    wrapper.set_base_optimizer(optimizer.base.AdamOptimizer(lr, epsilon=1e-3))
    wrapper.append_grad_modifier(optimizer.grad_modifier.LearningRateMultiplier([
        ('*/b', 2.0),
    ]))

    # To make the training more stable, we use grad clip by value.
    wrapper.append_grad_modifier(optimizer.grad_modifier.GlobalGradClip(0.001))
    # wrapper.append_grad_modifier(optimizer.grad_modifier.GlobalGradClipByAvgNorm(0.1))
    env.set_optimizer(wrapper)


def make_dataflow_train(env):
    batch_size = get_env('trainer.batch_size')

    df = flow.QueueDataFlow(env.data_queue)
    df = flow.BatchDataFlow(df, batch_size, sample_dict={
        'state': np.empty((batch_size, ) + get_input_shape(), dtype='float32'),
        'action': np.empty((batch_size, ) + get_action_shape(), dtype='int64'),
        'future_reward': np.empty((batch_size, ), dtype='float32')
    })
    return df


@cached_result
def get_action_shape():
    p = make_player()
    n = p.action_space.shape
    del p
    return tuple(n)


@cached_result
def get_action_range():
    p = make_player()
    l, h = p.action_space.low, p.action_space.high
    del p

    # Convert it to float32 to match the network's data type.
    return l.astype('float32'), h.astype('float32')


@cached_result
def get_input_shape():
    p = make_player()
    p.restart()
    input_shape = p.current_state.shape
    del p

    return input_shape


def sample_action(policy):
    space = get_env('a3c.actor_space')
    action = []
    for i, s in enumerate(space):
        a = random.choice(len(s), p=policy[i])
        action.append(a)
    return action


def player_func(pid, requester):
    player = make_player()
    actor_space = get_env('a3c.actor_space')
    player.restart()
    state = player.current_state
    reward = 0
    is_over = False
    with requester.activate():
        while True:
            action = requester.query('data', (state, reward, is_over))
            mapped_action = actor_space[np.arange(len(action)), action]
            reward, is_over = player.action(mapped_action)

            if len(player.stats['score']) > 0:
                score = player.stats['score'][-1]
                requester.query('stat', {'async/train/score': score}, do_recv=False)
                player.clear_stats()
            state = player.current_state


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

        out = func(state=batched_state[:nr_total])
        for i in range(nr_total):
            if is_inference:
                action = out['policy'][i]
            else:
                action = sample_action(out['policy_explore'][i])

            callbacks[i](action, out['value'][i])


def make_a3c_configs(env):
    from common_a3c import on_data_func, on_stat_func
    predictor_func = functools.partial(_predictor_func, is_inference=False)

    env.player_master.player_func = player_func
    env.player_master.predictor_func = predictor_func
    env.player_master.on_data_func = on_data_func
    env.player_master.on_stat_func = on_stat_func

    env.players_history = collections.defaultdict(list)


def main_train(trainer):
    from tartist.plugins.trainer_enhancer import summary
    summary.enable_summary_history(trainer, extra_summary_types={
        'async/train/score': 'async_scalar',
        'async/inference/score': 'async_scalar',
    })
    summary.enable_echo_summary_scalar(trainer, summary_spec={
        'async/train/score': ['avg', 'max'],
        'async/inference/score': ['avg', 'max']
    })

    from tartist.plugins.trainer_enhancer import progress
    progress.enable_epoch_progress(trainer)

    from tartist.plugins.trainer_enhancer import snapshot
    snapshot.enable_snapshot_saver(trainer, save_interval=5)

    from tartist.core import register_event
    from common_a3c import main_inference_play_multithread

    def on_epoch_after(trainer):
        if trainer.epoch > 0 and trainer.epoch % 2 == 0:
            main_inference_play_multithread(trainer, make_player=make_player)

    # This one should run before monitor.
    register_event(trainer, 'epoch:after', on_epoch_after, priority=5)

    trainer.train()


def main_demo(env, func):
    dump_dir = get_env('dir.demo', os.path.join(get_env('dir.root'), 'demo'))
    logger.info('Demo dump dir: {}'.format(dump_dir))
    player = make_player(is_train=False, dump_dir=dump_dir)
    repeat_time = get_env('a3c.demo.nr_plays', 1)

    def get_action(inp, func=func):
        action = func(state=inp[np.newaxis])['policy'][0].argmax()
        return action

    for i in range(repeat_time):
        player.play_one_episode(get_action)
        logger.info('#{} play score={}'.format(i, player.stats['score'][-1]))
