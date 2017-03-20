# -*- coding:utf8 -*-
# File   : desc_a3c_atari.py
# Author : Jiayuan Mao
#          Honghua Dong
# Email  : maojiayuan@gmail.com
#          dhh19951@gmail,com
# Date   : 3/18/17
#
# This file is part of TensorArtist

import threading
import collections
import numpy as np
import tensorflow as tf

from tartist.core import get_env, get_logger
from tartist.core.utils.cache import cached_result
from tartist.core.utils.naming import get_dump_directory, get_data_directory
from tartist.data import flow
from tartist.nn import opr as O, optimizer, summary, train
from tartist.nn.train.gan import GANGraphKeys
from tartist import rl, random, image


logger = get_logger(__file__)

__envs__ = {
    'dir': {
        'root': get_dump_directory(__file__),
        'data': get_data_directory('WellKnown/mnist')
    },

    'a3c': {
        'env_name': 'Breakout-v0',
        'frame_history': 4,
        'limit_length': 40000,
        'max_time': 5,
        'nr_players': 2,
        'nr_predictors': 2,
        'gamma': 0.99,
        'predictor': {
            'outputs_name': ['value', 'policy']
        }
    },

    'dataset': {
        'input_shape': (84, 84)
    },

    'trainer': {
        'learning_rate': 2e-4,

        'batch_size': 8,
        'epoch_size': 100,
        'nr_epochs': 200,

        'gamma': 0.99,

        'env_flags': {
            'log_device_placement': False
        }
    }
}

__trainer_cls__ = rl.train.A3CTrainer
__trainer_env_cls__ = rl.train.A3CTrainerEnv


def make_network(env):
    with env.create_network() as net:
        input_shape = get_env('dataset.input_shape')
        frame_history = get_env('a3c.frame_history')
        h, w, c = input_shape[0], input_shape[1], 3 * frame_history

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

        logits = O.softmax(policy, name='logits')
        value = value.remove_axis(1, name='value')

        expf = O.scalar('explore_factor', 1, trainable=False)
        policy = O.softmax(policy * expf, name='policy')

        net.add_output(policy, name='policy')
        net.add_output(value, name='value')

        if env.phase is env.Phase.TRAIN:
            action = O.placeholder('action', shape=(None, ), dtype=tf.int64)
            future_reward = O.placeholder('future_reward', shape=(None, ))

            log_logits = O.log(logits + 1e-6)
            log_pi_a_given_s = (log_logits * tf.one_hot(action, get_player_nr_actions())).sum(axis=1)
            advantage = future_reward - O.zero_grad(value, name='advantage')
            policy_cost = (log_pi_a_given_s * advantage).mean(name='policy_cost')
            xentropy_cost = (-logits * log_logits).mean(name='xentropy_cost')
            value_loss = O.raw_l2_loss('raw_value_loss', future_reward, value).mean(name='value_loss')
            # value_loss = O.truediv(value_loss, future_reward.shape[0].astype('float32'), name='value_loss')
            entropy_beta = O.scalar('entropy_beta', 0.01, trainable=False)
            loss = tf.add_n([-policy_cost, -xentropy_cost * entropy_beta, value_loss], name='loss')

            net.set_loss(loss)

            for v in [policy_cost, xentropy_cost, value_loss, 
                      value.mean(name='predict_value'), advantage.rms(name='rms_advantage'), loss]:
                summary.scalar(v)


def make_player():
    def resize_state(s):
        return image.resize(s, get_env('dataset.input_shape'))

    p = rl.GymRLEnviron(get_env('a3c.env_name'))
    p = rl.MapStateProxyRLEnviron(p, resize_state)
    p = rl.GymHistoryProxyRLEnviron(p, get_env('a3c.frame_history'))
    p = rl.LimitLengthProxyRLEnviron(p, get_env('a3c.limit_length'))
    return p


@cached_result
def get_player_nr_actions():
    p = make_player()
    n = p.action_space.nr_actions
    del p
    return n


def player_func(i, requester):
    player = make_player()
    player.restart()
    state = player.current_state
    reward = 0
    is_over = False
    with requester.activate():
        while True:
            # must have an action field
            action = requester.query({
                'type': 'data',
                'state': state,
                'reward': reward,
                'is_over': is_over
            })
            reward, is_over = player.action(action)
            if is_over:
                _ = requester.query({
                    'type': 'stat',
                    'score': player.stats['score'][-1]
                })
                player.clear_stats()
            state = player.current_state


def predictor_func(i, queue, func):
    while True:
        inp, callback = queue.get()
        out = func(state=inp['state'][np.newaxis])
        policy = out['policy'][0]
        action = random.choice(len(policy), p=policy)
        callback({
            'type': 'data-rep',
            'action': action,
            'value': out['value'][0]
        })


PlayerHistory = collections.namedtuple('PlayerHistory', ('state', 'action', 'value', 'reward'))


def on_data_func(env, player_router, identifier, inp_data):
    predictor_queue = env.predictors_queue
    data_queue = env.data_queue
    player_history = env.players_history[identifier]

    def parse_history(history, is_over):
        num = len(history)
        if is_over:
            r = 0
            env.players_history[identifier] = []
        elif num == get_env('a3c.max_time') + 1:
            history, last = history[:-1], history[-1]
            r = last.reward
            env.players_history[identifier] = [last]
        else:
            return

        gamma = get_env('a3c.gamma')
        for i in history[::-1]:
            r = np.clip(i.reward, -1, 1) + gamma * r
            data_queue.put_nowait({'state': i.state, 'action': i.action, 'future_reward': r})

    def callback(out_data):
        state = inp_data['state']
        predict_value, action = out_data['value'], out_data['action']
        player_router.send(identifier, action)
        with env.players_history_lock:
            player_history.append(PlayerHistory(state, action, predict_value, None))

    if len(player_history) > 0:
        with env.players_history_lock:
            last, reward = player_history[-1], inp_data['reward']
            is_over = inp_data['is_over']
            player_history[-1] = PlayerHistory(last[0], last[1], last[2], reward)
            parse_history(player_history, is_over)

    predictor_queue.put((inp_data, callback))


def on_stat_func(env, inp_data):
    if env.owner_trainer is not None:
        mgr = env.owner_trainer.runtime.get('summary_history', None)
        if mgr is not None:
            mgr.put_async_scalar('async/score', inp_data['score'])


def make_a3c_configs(env):
    env.player_func = player_func
    env.predictor_func = predictor_func
    env.on_data_func = on_data_func
    env.on_stat_func = on_stat_func
    env.players_history = collections.defaultdict(list)
    env.players_history_lock = threading.Lock()


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
    input_shape = get_env('dataset.input_shape')
    frame_history = get_env('a3c.frame_history')
    h, w, c = input_shape[0], input_shape[1], 3 * frame_history

    df = flow.QueueDataFlow(env.data_queue)
    df = flow.BatchDataFlow(df, batch_size, sample_dict={
        'state': np.empty((batch_size, h, w, c), dtype='float32'),
        'action': np.empty((batch_size, ), dtype='int32'),
        'future_reward': np.empty((batch_size, ), dtype='float32')
    })
    return df


def main_train(trainer):
    from tartist.plugins.trainer_enhancer import summary
    summary.enable_summary_history(trainer, extra_summary_types={'async/score': 'async_scalar'})
    summary.enable_echo_summary_scalar(trainer)

    from tartist.plugins.trainer_enhancer import progress
    progress.enable_epoch_progress(trainer)

    from tartist.plugins.trainer_enhancer import snapshot
    snapshot.enable_snapshot_saver(trainer)

    trainer.train()

