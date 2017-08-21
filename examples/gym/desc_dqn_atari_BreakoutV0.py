# -*- coding:utf8 -*-
# File   : desc_dqn_roboschool_HopperV1.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 15/08/2017
# 
# This file is part of TensorArtist.

import os
import threading
import roboschool

import numpy as np

from tartist import image
from tartist.app import rl
from tartist.core import get_env, get_logger
from tartist.core.utils.cache import cached_result
from tartist.core.utils.meta import map_exec
from tartist.core.utils.naming import get_dump_directory
from tartist.nn import opr as O, optimizer, summary

assert roboschool

logger = get_logger(__file__)

__envs__ = {
    'dir': {
        'root': get_dump_directory(__file__),
    },
    'dqn': {
        'env_name': 'Breakout-v0',
        'input_shape': (84, 84),

        'nr_history_frames': 4,
        'max_nr_steps': 40000,

        # gamma and TD steps in future_reward
        'gamma': 0.99,
        'nr_td_steps': 1,

        'collector': {
            'target': 64000,
            'nr_workers': 4,
            'nr_predictors': 2,

            # Add 'value' if you don't use a linear value regressor.
            'predictor_output_names': ['max_q', 'argmax_q']
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
        'policy_learning_rate': 0.0003,
        'value_learning_rate': 0.001,
        'nr_epochs': 200,

        # Parameters for Q-learner.
        'batch_size': 64,
        'data_repeat': 10,
    }
}

__envs__['trainer']['epoch_size'] = (
    __envs__['dqn']['collector']['target'] // __envs__['trainer']['batch_size'] *
    __envs__['trainer']['data_repeat']
)

__trainer_cls__ = rl.train.PPOTrainer
__trainer_env_cls__ = rl.train.PPOTrainerEnv


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
        q_pred = O.fc('fcq', _, get_player_nr_actions())
        max_q = q_pred.max(axis=1)
        argmax_q = q_pred.argmax(axis=1)

        net.add_output(q_pred, name='q_pred')
        net.add_output(max_q, name='max_q')
        net.add_output(argmax_q, name='argmax_q')

        if is_train:
            q_label = O.placeholder('q_value', shape=(None, ), dtype='float32')
            q_loss = O.raw_l2_loss('raw_q_loss', q_pred, q_label).mean(name='q_loss')
            net.set_loss(q_loss)

    if is_train:
        env.set_slave_devices(slave_devices)


def make_player(is_train=True, dump_dir=None):
    def resize_state(s):
        return image.resize(s, get_env('dqn.input_shape'), interpolation='NEAREST')

    p = rl.GymRLEnviron(get_env('dqn.env_name'), dump_dir=dump_dir)
    p = rl.MapStateProxyRLEnviron(p, resize_state)
    p = rl.HistoryFrameProxyRLEnviron(p, get_env('dqn.nr_history_frames'))

    p = rl.LimitLengthProxyRLEnviron(p, get_env('dqn.max_nr_steps'))
    if not is_train:
        p = rl.GymPreventStuckProxyRLEnviron(p, get_env('dqn.inference.max_antistuck_repeat'), 1)
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
    def _outputs2action(outputs):
        return outputs['max_q']

    collector = rl.train.SynchronizedExperienceCollector(
        env, make_player, _outputs2action,
        nr_workers=get_env('dqn.collector.nr_workers'), nr_predictors=get_env('dqn.collector.nr_workers'),
        predictor_output_names=get_env('dqn.collector.predictor_output_names'),
        mode='EPISODE-STEP'
    )

    return rl.utils.QLearningDataFlow(collector, target=get_env('dqn.collector.target'),
                                      gamma=get_env('dqn.gamma'), nr_td_steps=get_env('dqn.nr_td_steps'),
                                      batch_size=get_env('trainer.batch_size'), nr_repeat=get_env('trainer.nr_repeat'))


@cached_result
def get_player_nr_actions():
    p = make_player()
    n = p.action_space.nr_actions
    del p
    return n


@cached_result
def get_input_shape():
    input_shape = get_env('dqn.input_shape')
    nr_history_frames = get_env('dqn.nr_history_frames')
    h, w, c = input_shape[0], input_shape[1], 3 * nr_history_frames
    return h, w, c


def main_inference_play_multithread(trainer):
    def runner():
        func = trainer.env.make_func()
        func.compile({'theta': trainer.env.network.outputs['theta']})
        player = make_player()
        score = player.evaluate_one_episode(func)

        mgr = trainer.runtime.get('summary_histories', None)
        if mgr is not None:
            mgr.put_async_scalar('inference/score', score)

    nr_players = get_env('dqn.inference.nr_plays')
    pool = [threading.Thread(target=runner) for _ in range(nr_players)]
    map_exec(threading.Thread.start, pool)
    map_exec(threading.Thread.join, pool)


def main_train(trainer):
    # Register plugins.
    from tartist.plugins.trainer_enhancer import summary
    summary.enable_summary_history(trainer, extra_summary_types={
        'inference/score': 'async_scalar',
    })
    summary.enable_echo_summary_scalar(trainer, summary_spec={
        'inference/score': ['avg', 'max']
    })

    from tartist.plugins.trainer_enhancer import progress
    progress.enable_epoch_progress(trainer)

    from tartist.plugins.trainer_enhancer import snapshot
    snapshot.enable_snapshot_saver(trainer, save_interval=2)

    def on_epoch_after(trainer):
        if trainer.epoch > 0 and trainer.epoch % 2 == 0:
            main_inference_play_multithread(trainer)

    # This one should run before monitor.
    trainer.register_event('epoch:after', on_epoch_after, priority=5)

    trainer.train()


def main_demo(env, func):
    func.compile({'theta': env.network.outputs['theta']})

    dump_dir = get_env('dir.demo', os.path.join(get_env('dir.root'), 'demo'))
    logger.info('Demo dump dir: {}'.format(dump_dir))
    player = make_player(dump_dir=dump_dir)
    repeat_time = get_env('dqn.demo.nr_plays', 1)

    def get_action(inp, func=func):
        action = func(state=inp[np.newaxis])['max_q'][0]
        return action

    for i in range(repeat_time):
        player.play_one_episode(get_action)
        logger.info('#{} play score={}'.format(i, player.stats['score'][-1]))
