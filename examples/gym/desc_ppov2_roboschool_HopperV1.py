# -*- coding:utf8 -*-
# File   : desc_ppov2_roboschool_HopperV1.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 15/08/2017
# 
# This file is part of TensorArtist.

import os
import threading
import roboschool

import numpy as np

from tartist.app import rl
from tartist.core import get_env, get_logger
from tartist.core.utils.cache import cached_result
from tartist.core.utils.g import g
from tartist.core.utils.meta import map_exec
from tartist.core.utils.naming import get_dump_directory
from tartist.nn import opr as O, optimizer, summary

assert roboschool

logger = get_logger(__file__)

__envs__ = {
    'dir': {
        'root': get_dump_directory(__file__),
    },
    'ppo': {
        'env_name': 'RoboschoolHopper-v1',
        'max_nr_steps': 2000,

        'gamma': 0.99,
        'gae': {
            'lambda': 0.95,
        },

        'epsilon': 0.2,

        'collector': {
            'target': 25000,
            'nr_workers': 4,
            'nr_predictors': 2,

            'predictor_output_names': ['policy', 'theta', 'value']
        },
        'inference': {
            'nr_plays': 20
        },
        'demo': {
            'nr_plays': 20
        },
   },
    'trainer': {
        'learning_rate': 0.0003,
        'epoch_size': 5,
        'nr_epochs': 200,

        # Parameters for PPO optimizer.
        'batch_size': 64,
        'data_repeat': 10,
    }
}

__trainer_cls__ = rl.train.PPOTrainerV2
__trainer_env_cls__ = rl.train.PPOTrainerEnvV2

g.entropy_beta = 0.


def make_network(env):
    with env.create_network() as net:
        net.dist = O.distrib.GaussianDistribution('policy', size=get_action_shape()[0], fixed_std=False)

        state = O.placeholder('state', shape=(None, ) + get_input_shape())
        batch_size = state.shape[0]

        # We have to define variable scope here for later optimization.

        with env.variable_scope('policy'):
            _ = state
 
            _ = O.fc('fc1', _, 64, nonlin=O.relu)
            _ = O.fc('fc2', _, 64, nonlin=O.relu)
            mu = O.fc('fc_mu', _, net.dist.sample_size, nonlin=O.tanh)
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
            entropy_beta = O.scalar('entropy_beta', g.entropy_beta)

            log_prob = net.dist.log_likelihood(action, theta, process_theta=True)
            log_prob_old = net.dist.log_likelihood(action, theta_old, process_theta=True)

            ratio = O.exp(log_prob - log_prob_old)
            epsilon = get_env('ppo.epsilon')
            surr1 = ratio * advantage # surrogate from conservative policy iteration
            surr2 = O.clip_by_value(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantage
            policy_loss = -O.reduce_mean(O.min(surr1, surr2))  # PPO's pessimistic surrogate (L^CLIP)
            entropy = net.dist.entropy(theta, process_theta=True).mean()
            entropy_loss = -entropy_beta * entropy

            net.add_output(policy_loss, name='policy_loss')
            net.add_output(entropy_loss, name='entropy_loss')

            summary.scalar('policy_entropy', entropy)

        with env.variable_scope('value'):
            _ = state
            _ = O.fc('fc1', _, 64, nonlin=O.relu)
            _ = O.fc('fc2', _, 64, nonlin=O.relu)
            value = O.fc('fcv', _, 1)
            value = value.remove_axis(1)
            net.add_output(value, name='value')

        if env.phase == env.Phase.TRAIN:
            value_label = O.placeholder('value_label', shape=(None, ))
            value_old = O.placeholder('value_old', shape=(None, ))

            value_surr1 = O.raw_l2_loss('raw_value_loss_surr1', value, value_label)
            value_clipped = value_old + O.clip_by_value(value - value_old, -epsilon, epsilon)
            value_surr2 = O.raw_l2_loss('raw_value_loss_surr2', value_clipped, value_label)
            value_loss = O.reduce_mean(O.max(value_surr1, value_surr2))
            net.add_output(value_loss, name='value_loss')

        if env.phase == env.Phase.TRAIN:
            loss = O.identity(policy_loss + entropy_loss + value_loss, name='total_loss')
            net.set_loss(loss)


def make_player(dump_dir=None):
    p = rl.GymRLEnviron(get_env('ppo.env_name'), dump_dir=dump_dir)
    p = rl.LimitLengthProxyRLEnviron(p, get_env('ppo.max_nr_steps'))
    return p


def make_optimizer(env):
    wrapper = optimizer.OptimizerWrapper()
    wrapper.set_base_optimizer(optimizer.base.AdamOptimizer(get_env('trainer.learning_rate'), epsilon=1e-3))
    wrapper.append_grad_modifier(optimizer.grad_modifier.LearningRateMultiplier([
        ('*/b', 2.0),
    ]))
    env.set_optimizer(wrapper)


def make_dataflow_train(env):
    def _outputs2action(outputs):
        return outputs['policy']

    collector = rl.train.SynchronizedExperienceCollector(
        env, make_player, _outputs2action,
        nr_workers=get_env('ppo.collector.nr_workers'), nr_predictors=get_env('ppo.collector.nr_workers'),
        predictor_output_names=get_env('ppo.collector.predictor_output_names'),
        mode='EPISODE-STEP'
    )

    return rl.train.SynchronizedTrajectoryDataFlow(
        collector, target=get_env('ppo.collector.target'), incl_value=True)


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

    nr_players = get_env('ppo.inference.nr_plays')
    pool = [threading.Thread(target=runner) for _ in range(nr_players)]
    map_exec(threading.Thread.start, pool)
    map_exec(threading.Thread.join, pool)


def main_train(trainer):
    from tartist.app.rl.utils.adv import GAEComputer
    from tartist.random.sampler import SimpleBatchSampler
    trainer.set_adv_computer(GAEComputer(get_env('ppo.gamma'), get_env('ppo.gae.lambda')))
    trainer.set_batch_sampler(SimpleBatchSampler(get_env('trainer.batch_size'), get_env('trainer.data_repeat')))

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
    snapshot.enable_snapshot_saver(trainer, save_interval=1)

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
    repeat_time = get_env('ppo.demo.nr_plays', 1)

    def get_action(inp, func=func):
        policy = func(state=inp[np.newaxis])['theta'][0]
        return _theta2action(policy)

    for i in range(repeat_time):
        player.play_one_episode(get_action)
        logger.info('#{} play score={}'.format(i, player.stats['score'][-1]))
