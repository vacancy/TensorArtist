# -*- coding:utf8 -*-
# File   : desc_a3c_atari.py
# Author : Jiayuan Mao
#          Honghua Dong
# Email  : maojiayuan@gmail.com
#          dhh19951@gmail,com
# Date   : 3/18/17
#
# This file is part of TensorArtist

import time
import queue
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
        'nr_players': 50,
        'nr_predictors': 2,
        'gamma': 0.99,
        'predictor': {
            'batch_size': 16,
            'outputs_name': ['value', 'policy']
        }
    },

    'dataset': {
        'input_shape': (84, 84)
    },

    'trainer': {
        'learning_rate': 0.001,

        'batch_size': 128,
        'epoch_size': 100,
        'nr_epochs': 200,

        'gamma': 0.99,

        'env_flags': {
            'log_device_placement': False
        }
    },
    'demo': {
        'customized': True,
        'repeat': 5
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

        logits = O.softmax(policy, name='logits')
        value = value.remove_axis(1, name='value')

        expf = O.scalar('explore_factor', 1, trainable=False)
        policy = O.softmax(policy * expf, name='policy')

        net.add_output(logits, name='logits')
        net.add_output(policy, name='policy')
        net.add_output(value, name='value')

        if env.phase is env.Phase.TRAIN:
            action = O.placeholder('action', shape=(None, ), dtype=tf.int64)
            future_reward = O.placeholder('future_reward', shape=(None, ))

            log_logits = O.log(logits + 1e-6)
            log_pi_a_given_s = (log_logits * tf.one_hot(action, get_player_nr_actions())).sum(axis=1)
            advantage = future_reward - O.zero_grad(value, name='advantage')
            policy_cost = (log_pi_a_given_s * advantage).mean(name='policy_cost')
            xentropy_cost = (-logits * log_logits).sum(axis=1).mean(name='xentropy_cost')
            value_loss = O.raw_l2_loss('raw_value_loss', future_reward, value).mean(name='value_loss')
            # value_loss = O.truediv(value_loss, future_reward.shape[0].astype('float32'), name='value_loss')
            entropy_beta = O.scalar('entropy_beta', 0.01, trainable=False)
            loss = tf.add_n([-policy_cost, -xentropy_cost * entropy_beta, value_loss], name='loss')

            net.set_loss(loss)

            for v in [policy_cost, xentropy_cost, value_loss, 
                      value.mean(name='predict_value'), advantage.rms(name='rms_advantage'), loss]:
                summary.scalar(v)

    if is_train:
        env.set_slave_devices(slave_devices)


def make_player(is_train=True):
    def resize_state(s):
        return image.resize(s, get_env('dataset.input_shape'))

    p = rl.GymRLEnviron(get_env('a3c.env_name'))
    p = rl.MapStateProxyRLEnviron(p, resize_state)
    p = rl.GymHistoryProxyRLEnviron(p, get_env('a3c.frame_history'))
    if is_train:
        p = rl.LimitLengthProxyRLEnviron(p, get_env('a3c.limit_length'))
    return p


@cached_result
def get_player_nr_actions():
    p = make_player()
    n = p.action_space.nr_actions
    del p
    return n


@cached_result
def get_input_shape():
    input_shape = get_env('dataset.input_shape')
    frame_history = get_env('a3c.frame_history')
    h, w, c = input_shape[0], input_shape[1], 3 * frame_history
    return h, w, c


def player_func(i, requester):
    player = make_player()
    player.restart()
    state = player.current_state
    reward = 0
    is_over = False
    with requester.activate():
        while True:
            # must have an action field
            action = requester.query('data', (state, reward, is_over))
            reward, is_over = player.action(action)
            
            if len(player.stats['score']) > 0:
                score = player.stats['score'][-1]
                requester.query('stat', (score, ), do_recv=False)
                player.clear_stats()
            state = player.current_state


def predictor_func(i, router, queue, func):
    batch_size = get_env('a3c.predictor.batch_size')
    while True:
        batched_state = np.empty((batch_size, ) + get_input_shape(), dtype='float32')
        callbacks = []
        for i in range(batch_size):
            identifier, inp, callback = queue.get()
            batched_state[i] = inp[0]
            callbacks.append(callback)
        out = func(state=batched_state)
        actions = []
        for i in range(batch_size):
            policy = out['policy'][i]
            action = random.choice(len(policy), p=policy)
            actions.append(action)

        for i in range(batch_size):
            callbacks[i](actions[i], out['value'][i])


PlayerHistory = collections.namedtuple('PlayerHistory', ('state', 'action', 'value', 'reward'))


def on_data_func(env, player_router, identifier, inp_data):
    predictor_queue = env.predictors_queue
    data_queue = env.data_queue
    player_history = env.players_history[identifier]

    state, reward, is_over = inp_data

    def parse_history(history, is_over):
        num = len(history)
        if is_over:
            r = 0
            env.players_history[identifier] = []
        elif num == get_env('a3c.max_time') + 1:
            history, last = history[:-1], history[-1]
            r = last.value
            env.players_history[identifier] = [last]
        else:
            return

        gamma = get_env('a3c.gamma')
        for i in history[::-1]:
            r = np.clip(i.reward, -1, 1) + gamma * r
            try:
                data_queue.put_nowait({'state': i.state, 'action': i.action, 'future_reward': r})
            except queue.Full:
                pass

    def callback(action, predict_value):
        player_router.send(identifier, action)
        player_history.append(PlayerHistory(state, action, predict_value, None))

    predictor_queue.put((identifier, inp_data, callback))

    if len(player_history) > 0:
        last = player_history[-1]
        player_history[-1] = PlayerHistory(last[0], last[1], last[2], reward)
        parse_history(player_history, is_over)


def on_stat_func(env, inp_data):
    if env.owner_trainer is not None:
        mgr = env.owner_trainer.runtime.get('summary_histories', None)
        if mgr is not None:
            mgr.put_async_scalar('async/score', inp_data[0])


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


def multi_thread_inference(trainer):
    scorer = [0]
    score_lock  = threading.Lock()
    def runner(scorer):
        func = trainer.env.make_func()
        func.compile(trainer.env.network.outputs)
        player = make_player(is_train=False)
        def get_action(inp, func=func):
            action = func(**{'state':[[inp]]})['logits'][0].argmax()
            print(action)
            return action
        print(player)
        player.play_one_episode(get_action)
        print(player.stats['score'])
        with score_lock:
            from IPython import embed; embed()
            scorer[0] += player.stats['score']
    num_runner = get_env('inference.runner', 1)
    threads_pool = []
    for i in range(num_runner):
        p = threading.Thread(target=runner, args=(scorer))
        p.start()
        threads_pool.append(p)
    for p in threads_pool:
        p.join()
    print('avg_score', scorer[0] / float(num_runner))


def main_train(trainer):
    from tartist.plugins.trainer_enhancer import summary
    summary.enable_summary_history(trainer, extra_summary_types={'async/score': 'async_scalar'})
    summary.enable_echo_summary_scalar(trainer)

    from tartist.plugins.trainer_enhancer import progress
    progress.enable_epoch_progress(trainer)

    from tartist.plugins.trainer_enhancer import snapshot
    snapshot.enable_snapshot_saver(trainer)

    from tartist.core import register_event
    def on_epoch_after(trainer):
        if trainer.epoch > 0 and trainer.epoch % get_env('inference.test_epochs', 2) == 0:
            multi_thread_inference(trainer)

    register_event(trainer, 'epoch:after', on_epoch_after)

    trainer.train()


def main_demo(env, func):
    player = make_player(is_train=False)
    repeat_time = get_env('demo.repeat', 1)
    def get_action(inp, func=func):
        action = func(**{'state':[[inp]]})['logits'][0].argmax()
        return action
    for i in range(repeat_time):
        if i != 0:
            player.restart()
        player.play_one_episode(get_action)
        print(i, player.stats['score'][-1])
