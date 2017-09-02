# -*- coding:utf8 -*-
# File   : desc_hp_a3c_atari_BreakoutV0.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/07/2017
# 
# This file is part of TensorArtist.

import collections
import functools
import os
import queue

import numpy as np
import libhpref

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

        'nr_history_frames': 4,
        'max_nr_steps': 40000,

        'gamma': 0.99,
        'nr_td_steps': 5,

        'nr_players': 2,
        'nr_predictors': 2,
        'predictor': {
            'batch_size': 1,
            'outputs_name': ['value', 'policy_explore']
        },

        'inference': {
            'nr_plays': 20,
            'max_antistuck_repeat': 30
        },
        'demo': {
            'nr_plays': 5
        }
    },

    'trainer': {
        'learning_rate': 0.001,

        'batch_size': 128,
        'epoch_size': 100,
        'nr_epochs': 200,
    },

    'rpredictor': {
        'nr_ensembles': 3,
        'learning_rate': 0.001,
        'batch_size': 16,
        'epoch_size': 20,
        'nr_epochs': 100,
        'retrain_thresh': 25,
    },

    'pcollector': {
        'video_length': 25,
        'window_length': 25,
        'pool_size': 25,
        'web_configs': {
            'title': 'RL Human Preference Collector',
            'author': 'TensorArtist authors',
            'port': 8888,
        }
    }
}

__trainer_cls__ = libhpref.HPA3CTrainer
__trainer_env_cls__ = libhpref.HPA3CTrainerEnv


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

        if is_train:
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


def make_rpredictor_network(env):
    is_train = env.phase is env.Phase.TRAIN

    with env.create_network() as net:
        h, w, c = get_input_shape()
        # Hack(MJY):: forced RGB input (instead of combination of history frames)
        c = 3

        dpc = env.create_dpcontroller()
        with dpc.activate():
            def inputs():
                state = O.placeholder('state', shape=(None, h, w, c))
                t1_state = O.placeholder('t1_state', shape=(None, h, w, c))
                t2_state = O.placeholder('t2_state', shape=(None, h, w, c))
                return [state, t1_state, t2_state]

            @O.auto_reuse
            def forward_conv(x):
                _ = x / 255.0
                with O.argscope(O.conv2d, nonlin=O.relu):
                    _ = O.conv2d('conv0', _, 32, 5)
                    _ = O.max_pooling2d('pool0', _, 2)
                    _ = O.conv2d('conv1', _, 32, 5)
                    _ = O.max_pooling2d('pool1', _, 2)
                    _ = O.conv2d('conv2', _, 64, 4)
                    _ = O.max_pooling2d('pool2', _, 2)
                    _ = O.conv2d('conv3', _, 64, 3)
                return _

            def forward(x, t1, t2):
                dpc.add_output(forward_conv(x), name='feature')
                dpc.add_output(forward_conv(t1), name='t1_feature')
                dpc.add_output(forward_conv(t2), name='t2_feature')

            dpc.set_input_maker(inputs).set_forward_func(forward)

        @O.auto_reuse
        def forward_fc(feature, action):
            action = O.one_hot(action, get_player_nr_actions())
            _ = O.concat([feature.flatten2(), action], axis=1)
            _ = O.fc('fc0', _, 512, nonlin=O.p_relu)
            reward = O.fc('fc_reward', _, 1)
            return reward

        action = O.placeholder('action', shape=(None, ), dtype='int64')
        net.add_output(forward_fc(dpc.outputs['feature'], action), name='reward')

        if is_train:
            t1_action = O.placeholder('t1_action', shape=(None, ), dtype='int64')
            t1_reward_exp = O.exp(forward_fc(dpc.outputs['t1_feature'], t1_action).sum())
            t2_action = O.placeholder('t2_action', shape=(None, ), dtype='int64')
            t2_reward_exp = O.exp(forward_fc(dpc.outputs['t2_feature'], t2_action).sum())

            pref = O.placeholder('pref')
            p1, p2 = 1 - pref, pref

            p_greater = t1_reward_exp / (t1_reward_exp + t2_reward_exp)
            loss = - p1 * O.log(p_greater) - p2 * O.log(1 - p_greater)

            net.set_loss(loss)


def make_player(is_train=True, dump_dir=None):
    def resize_state(s):
        return image.resize(s, get_env('a3c.input_shape'), interpolation='NEAREST')

    p = rl.GymRLEnviron(get_env('a3c.env_name'), dump_dir=dump_dir)
    p = rl.MapStateProxyRLEnviron(p, resize_state)
    p = rl.HistoryFrameProxyRLEnviron(p, get_env('a3c.nr_history_frames'))

    p = rl.LimitLengthProxyRLEnviron(p, get_env('a3c.max_nr_steps'))
    if is_train:
        p = rl.AutoRestartProxyRLEnviron(p)
    else:
        p = rl.GymPreventStuckProxyRLEnviron(p, get_env('a3c.inference.max_antistuck_repeat'), 1)
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


def make_rpredictor_optimizer(env):
    wrapper = optimizer.OptimizerWrapper()
    wrapper.set_base_optimizer(optimizer.base.AdamOptimizer(get_env('rpredictor.learning_rate'), epsilon=1e-3))
    wrapper.append_grad_modifier(optimizer.grad_modifier.LearningRateMultiplier([
        ('*/b', 2.0),
    ]))
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
    nr_history_frames = get_env('a3c.nr_history_frames')
    h, w, c = input_shape[0], input_shape[1], 3 * nr_history_frames
    return h, w, c


# HACK(MJY):: select the last 3 channels representing the current state (ignoring history frames).
def get_unproxied_state(state):
    if len(state.shape) == 3:
        return state[:, :, -3:]
    else:
        return state[:, :, :, -3:]


PlayerHistory = collections.namedtuple('PlayerHistory', ('state', 'action', 'value', 'reward', 'reward_var'))


def on_data_func(env, identifier, inp_data):
    router, task_queue = env.player_master.router, env.player_master.queue
    data_queue = env.data_queue
    player_history = env.players_history[identifier]

    state, is_over = inp_data

    def parse_history(history):
        num = len(history)
        if is_over:
            r = 0
            env.players_history[identifier] = []
        elif num == get_env('a3c.nr_td_steps') + 1:
            history, last = history[:-1], history[-1]
            r = last.value
            env.players_history[identifier] = [last]
        else:
            return

        gamma = get_env('a3c.gamma')
        for i in history[::-1]:
            r = np.clip(i.reward, -1, 1) + gamma * r
            # r = i.reward + gamma * r
            data_queue.put({'state': i.state, 'action': i.action, 'future_reward': r})

    def callback(action, reward_info, predict_value):
        # reward_info is of form (reward, reward_variance)
        player_history.append(PlayerHistory(state, action, predict_value, reward_info[0], reward_info[1]))
        # simply use state as observation
        unproxied_state = get_unproxied_state(state)
        env.pcollector.post_state(identifier, unproxied_state, unproxied_state, action, reward_info[1])
        router.send(identifier, (action, reward_info[0]))

    if len(player_history) > 0:
        parse_history(player_history)

    task_queue.put((identifier, inp_data, callback))


def on_stat_func(env, identifier, inp_data):
    if env.owner_trainer is not None:
        mgr = env.owner_trainer.runtime.get('summary_histories', None)
        if mgr is not None:
            for k, v in inp_data.items():
                mgr.put_async_scalar(k, v)


def player_func(pid, requester):
    player = make_player()
    player.restart()
    state = player.current_state
    is_over = False

    with requester.activate():
        while True:
            action, reward = requester.query('data', (state, is_over))
            _, is_over = player.action(action)

            if len(player.stats['score']) > 0:
                score = player.stats['score'][-1]
                requester.query('stat', {'async/train/score': score}, do_recv=False)
                player.clear_stats()

            state = player.current_state


def _predictor_func(pid, rpredictor, router, task_queue, func, is_inference=False):
    batch_size = get_env('a3c.predictor.batch_size')

    batched_state = np.empty((batch_size, ) + get_input_shape(), dtype='float32')
    # assume discrete action space
    batched_action = np.empty((batch_size, ), dtype='int32')

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
            policy = out['policy_explore'][i]
            if is_inference:
                # During inference, policy should be out['policy'][i].
                # However, these two are equivalent under argmax operation,
                # so we can directly use 'policy_explore' in output.
                action = policy.argmax()
            else:
                action = random.choice(len(policy), p=policy)

            batched_action[i] = action

        rewards = rpredictor.predict_batch(
            get_unproxied_state(get_unproxied_state(batched_state[:nr_total])),
            batched_action[:nr_total], ret_variance=True)

        for i in range(nr_total):
            callbacks[i](batched_action[i], rewards[i], out['value'][i])


def make_a3c_configs(env):
    predictor_func = functools.partial(_predictor_func, is_inference=False)

    env.player_master.player_func = player_func
    env.player_master.predictor_func = predictor_func
    env.player_master.on_data_func = on_data_func
    env.player_master.on_stat_func = on_stat_func

    predictor_desc = libhpref.PredictorDesc(make_rpredictor_network, make_rpredictor_optimizer,
                                            None, None, None)
    scheduler = libhpref.ExponentialDecayCollectorScheduler(1000, 200, get_env('trainer.nr_epochs'))
    env.player_master.rpredictor = rpredictor = libhpref.EnsemblePredictor(
        env, scheduler, predictor_desc,
        nr_ensembles=get_env('rpredictor.nr_ensembles'),
        devices=[env.master_device] * get_env('rpredictor.nr_ensembles'),
        nr_epochs=get_env('rpredictor.nr_epochs'), epoch_size=get_env('rpredictor.epoch_size'),
        retrain_thresh=get_env('rpredictor.retrain_thresh'))
    env.set_pcollector(libhpref.PreferenceCollector(
        rpredictor, get_env('pcollector.web_configs'),
        video_length=get_env('pcollector.video_length'), window_length=get_env('pcollector.window_length'),
        pool_size=get_env('pcollector.pool_size')))

    env.players_history = collections.defaultdict(list)


def main_train(trainer):
    from tartist.plugins.trainer_enhancer import summary
    summary.enable_summary_history(trainer, extra_summary_types={
        'async/train/score': 'async_scalar',
        'async/inference/score': 'async_scalar'
    })
    summary.enable_echo_summary_scalar(trainer, summary_spec={
        'async/train/score': ['avg', 'max'],
        'async/inference/score': ['avg', 'max']
    })

    from tartist.plugins.trainer_enhancer import progress
    progress.enable_epoch_progress(trainer)

    from tartist.plugins.trainer_enhancer import snapshot
    snapshot.enable_snapshot_saver(trainer)

    from tartist.core import register_event
    from common_hp_a3c import main_inference_play_multithread

    def on_epoch_after(trainer):
        if trainer.epoch > 0 and trainer.epoch % 2 == 0:
            main_inference_play_multithread(trainer, make_player=make_player)

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
