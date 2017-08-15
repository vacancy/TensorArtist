# -*- coding:utf8 -*-
# File   : desc_trpo_classic_LunarLanderContinuousV2.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/08/2017
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
from tartist.nn import opr as O, optimizer, summary

logger = get_logger(__file__)

__envs__ = {
    'dir': {
        'root': get_dump_directory(__file__),
    },
    'trpo': {
        'env_name': 'LunarLanderContinuous-v2',
        'max_nr_steps': 200,

        'gamma': 0.99,
        'max_kl': 0.001,
        'cg': {
            'damping': 0.001
        },

        'use_linear_vr': True,

        'collector': {
            'target': 20,
            'nr_workers': 8,
            'nr_predictors': 2,

            # Add 'value' if you don't use a linear value regressor.
            'predictor_output_names': ['policy', 'theta']
        },
        'inference': {
            'nr_plays': 20
        },
        'demo': {
            'nr_plays': 20
        },
   },
    'trainer': {
        'value_learning_rate': 0.001,
        'epoch_size': 5,
        'nr_epochs': 200,
    }
}

__trainer_cls__ = rl.train.TRPOTrainer
__trainer_env_cls__ = rl.train.TRPOTrainerEnv


def make_network(env):
    use_linear_vr = get_env('trpo.use_linear_vr')

    with env.create_network() as net:
        net.dist = O.distrib.GaussianDistribution('policy', size=get_action_shape()[0], fixed_std=False)
        if use_linear_vr:
            from tartist.app.rl.math_utils import LinearValueRegressor
            net.value_regressor = LinearValueRegressor()

        state = O.placeholder('state', shape=(None, ) + get_input_shape())
        # state = O.moving_average(state)
        # state = O.clip_by_value(state, -10, 10)
        batch_size = state.shape[0]

        # We have to define variable scope here for later optimization.

        with env.variable_scope('policy'):
            _ = state

            with O.argscope(O.fc):
                _ = O.fc('fc1', _, 64, nonlin=O.relu)
                _ = O.fc('fc2', _, 64, nonlin=O.relu)
                mu = O.fc('fc_mu', _, net.dist.sample_size)
                logstd = O.variable('logstd', O.truncated_normal_initializer(stddev=0.01),
                                    shape=(net.dist.sample_size, ), trainable=True)

            logstd = O.tile(logstd.add_axis(0), [batch_size, 1])
            theta = O.concat([mu, logstd], axis=1)

            policy = net.dist.sample(batch_size=batch_size, theta=theta, process_theta=True)
            policy = O.clip_by_value(policy, -1, 1)

            net.add_output(theta, name='theta')
            net.add_output(policy, name='policy')

        if env.phase == env.Phase.TRAIN:
            theta_old = O.placeholder('theta_old', shape=(None, net.dist.param_size))
            action = O.placeholder('action', shape=(None, net.dist.sample_size))
            advantage = O.placeholder('advantage', shape=(None, ))

            log_prob = net.dist.log_likelihood(action, theta, process_theta=True)
            log_prob_old = net.dist.log_likelihood(action, theta_old, process_theta=True)

            # Importance sampling of surrogate loss (L in paper).
            ratio = O.exp(log_prob - log_prob_old)
            policy_loss = -O.reduce_mean(ratio * advantage)

            kl = net.dist.kl(theta_p=theta_old, theta_q=theta, process_theta=True).mean()
            kl_self = net.dist.kl(theta_p=O.zero_grad(theta), theta_q=theta, process_theta=True).mean()
            entropy = net.dist.entropy(theta, process_theta=True).mean()

            net.add_output(policy_loss, name='policy_loss')
            net.add_output(kl, name='kl')
            net.add_output(kl_self, name='kl_self')

            summary.scalar('policy_entropy', entropy, collections=[rl.train.TRPOGraphKeys.POLICY_SUMMARIES])

        if not use_linear_vr:
            with env.variable_scope('value'):
                value = O.fc('fcv', state, 1)
                net.add_output(value, name='value')

            if env.phase == env.Phase.TRAIN:
                value_label = O.placeholder('value_label', shape=(None, ))
                value_loss = O.raw_l2_loss('raw_value_loss', value, value_label).mean(name='value_loss')
                net.add_output(value_loss, name='value_loss')


def make_player(dump_dir=None):
    p = rl.GymRLEnviron(get_env('trpo.env_name'), dump_dir=dump_dir)
    p = rl.LimitLengthProxyRLEnviron(p, get_env('trpo.max_nr_steps'))
    return p


def make_optimizer(env):
    opt = rl.train.TRPOOptimizer(env, max_kl=get_env('trpo.max_kl'), cg_damping=get_env('trpo.cg.damping'))
    env.set_policy_optimizer(opt)

    use_linear_vr = get_env('trpo.use_linear_vr')
    if not use_linear_vr:
        wrapper = optimizer.OptimizerWrapper()
        wrapper.set_base_optimizer(optimizer.base.AdamOptimizer(get_env('trainer.value_learning_rate'), epsilon=1e-3))
        wrapper.append_grad_modifier(optimizer.grad_modifier.LearningRateMultiplier([
            ('*/b', 2.0),
        ]))
        env.set_value_optimizer(wrapper)


def make_dataflow_train(env):
    collector = rl.train.SynchronizedExperienceCollector(
        env, make_player, _outputs2action,
        nr_workers=8, nr_predictors=2,
        predictor_output_names=get_env('trpo.collector.predictor_output_names')
    )

    use_linear_vr = get_env('trpo.use_linear_vr')
    return rl.train.TRPODataFlow(collector,
                                 target=get_env('trpo.collector.target'),
                                 gamma=get_env('trpo.gamma'),
                                 compute_adv=not use_linear_vr)


@cached_result
def get_input_shape():
    p = make_player()
    p.restart()
    input_shape = p.current_state.shape
    del p

    return input_shape


@cached_result
def get_action_shape():
    p = make_player()
    n = p.action_space.shape
    del p
    return tuple(n)


def _outputs2action(outputs):
    return outputs['policy']


def _theta2action(theta):
    return np.clip(theta[:get_action_shape()[0]], -1, 1)


def _evaluate(player, func):
    score = 0
    player.restart()
    while True:
        policy = func(state=player.current_state[np.newaxis])['theta'][0]
        reward, done = player.action(_theta2action(policy))
        score += reward
        if done:
            player.finish()
            break
    return score


def main_inference_play_multithread(trainer):
    def runner():
        func = trainer.env.make_func()
        func.compile({'theta': trainer.env.network.outputs['theta']})
        player = make_player()
        score = _evaluate(player, func)

        mgr = trainer.runtime.get('summary_histories', None)
        if mgr is not None:
            mgr.put_async_scalar('inference/score', score)

    nr_players = get_env('trpo.inference.nr_plays')
    pool = [threading.Thread(target=runner) for _ in range(nr_players)]
    map_exec(threading.Thread.start, pool)
    map_exec(threading.Thread.join, pool)


def main_train(trainer):
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

    def on_epoch_after(trainer):
        if trainer.epoch > 0 and trainer.epoch % 2 == 0:
            main_inference_play_multithread(trainer)

    # this one should run before monitor
    trainer.register_event('epoch:after', on_epoch_after, priority=5)

    trainer.train()


def main_demo(env, func):
    func.compile({'theta': env.network.outputs['theta']})

    dump_dir = get_env('dir.demo', os.path.join(get_env('dir.root'), 'demo'))
    logger.info('Demo dump dir: {}'.format(dump_dir))
    player = make_player(dump_dir=dump_dir)
    repeat_time = get_env('trpo.demo.nr_plays', 1)

    def get_action(inp, func=func):
        policy = func(state=inp[np.newaxis])['theta'][0]
        return _theta2action(policy)

    for i in range(repeat_time):
        player.play_one_episode(get_action)
        logger.info('#{} play score={}'.format(i, player.stats['score'][-1]))
