# -*- coding:utf8 -*-
# File   : desc_cem_classic_CartPoleV0.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/08/2017
# 
# This file is part of TensorArtist.


import os
import threading
import numpy as np

from tartist.app import rl
from tartist.core import get_env, get_logger
from tartist.core.utils.cache import cached_result
from tartist.core.utils.naming import get_dump_directory
from tartist.core.utils.meta import map_exec
from tartist.nn import opr as O

logger = get_logger(__file__)

__envs__ = {
    'dir': {
        'root': get_dump_directory(__file__),
    },
    'cem': {
        'env_name': 'CartPole-v0',
        'top_frac': 0.2,
        'max_nr_steps': 200,

        'inference': {
            'nr_plays': 20
        },
        'demo': {
            'nr_plays': 20
        },
    },
    'trainer': {
        'epoch_size': 200,
        'nr_epochs': 50,
    }
}

__trainer_cls__ = rl.train.EvolutionBasedTrainer


def make_network(env):
    with env.create_network() as net:
        state = O.placeholder('state', shape=(None, ) + get_input_shape())
        logits = O.fc('fc', state, get_action_shape())
        net.add_output(logits, name='policy')


def make_player(dump_dir=None):
    p = rl.GymRLEnviron(get_env('cem.env_name'), dump_dir=dump_dir)
    p = rl.LimitLengthProxyRLEnviron(p, get_env('cem.max_nr_steps'))
    return p


def make_optimizer(env):
    optimizer = rl.train.CEMOptimizer(env, top_frac=get_env('cem.top_frac'))
    env.set_optimizer(optimizer)


@cached_result
def get_input_shape():
    p = make_player()
    p.restart()
    input_shape = p.current_state.shape
    del p

    return input_shape


@cached_result
def get_action_shape():
    return 1


def _policy2action(policy):
    return int(policy > 0)


def _evaluate(player, func):
    score = 0
    player.restart()
    while True:
        policy = func(state=player.current_state[np.newaxis])['policy'][0]
        reward, done = player.action(_policy2action(policy))
        score += reward
        if done:
            player.finish()
            break
    return score


def main_inference_play_multithread(trainer):
    def runner():
        func = trainer.env.make_func()
        func.compile(trainer.env.network.outputs)
        player = make_player()
        score = _evaluate(player, func)

        mgr = trainer.runtime.get('summary_histories', None)
        if mgr is not None:
            mgr.put_async_scalar('inference/score', score)

    nr_players = get_env('cem.inference.nr_plays')
    pool = [threading.Thread(target=runner) for _ in range(nr_players)]
    map_exec(threading.Thread.start, pool)
    map_exec(threading.Thread.join, pool)


def main_train(trainer):
    # Compose the evaluator
    player = make_player()

    def evaluate_train(trainer, p=player):
        return _evaluate(player=p, func=trainer.pred_func)

    trainer.set_evaluator(evaluate_train)

    # Register plugins
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
    snapshot.enable_snapshot_saver(trainer, save_interval=1)

    def on_epoch_before(trainer):
        v = max(5 - trainer.epoch / 10, 0)
        trainer.optimizer.param_std += v

    def on_epoch_after(trainer):
        if trainer.epoch > 0 and trainer.epoch % 2 == 0:
            main_inference_play_multithread(trainer)

    # this one should run before monitor
    trainer.register_event('epoch:before', on_epoch_before, priority=5)
    trainer.register_event('epoch:after', on_epoch_after, priority=5)

    trainer.train()


def main_demo(env, func):
    dump_dir = get_env('dir.demo', os.path.join(get_env('dir.root'), 'demo'))
    logger.info('Demo dump dir: {}'.format(dump_dir))
    player = make_player(dump_dir=dump_dir)
    repeat_time = get_env('cem.demo.nr_plays', 1)

    def get_action(inp, func=func):
        policy = func(state=inp[np.newaxis])['policy'][0]
        return _policy2action(policy)

    for i in range(repeat_time):
        player.play_one_episode(get_action)
        logger.info('#{} play score={}'.format(i, player.stats['score'][-1]))
