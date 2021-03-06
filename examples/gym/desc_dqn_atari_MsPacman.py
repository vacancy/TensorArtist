# -*- coding:utf8 -*-

import os
import functools
import threading

import numpy as np

from tartist import image, random
from tartist.app import rl
from tartist.core import get_env, get_logger
from tartist.core.utils.cache import cached_result
from tartist.core.utils.meta import map_exec
from tartist.core.utils.naming import get_dump_directory
from tartist.nn import opr as O, optimizer, summary

logger = get_logger(__file__)

__envs__ = {
    'dir': {
        'root': get_dump_directory(__file__),
    },
    'dqn': {
        'env_name': 'MsPacmanNoFrameskip-v4',
        'input_shape': (84, 84),

        'nr_history_frames': 4,
        'max_nr_steps': 40000,
        'frame_skip': 4,

        # gamma and TD steps in future_reward
        # nr_td_steps must be 1
        'gamma': 0.9,
        'nr_td_steps': 1,

        'expreplay': {
            'maxsize': 500000,
        },

        'collector': {
            'mode': 'EPISODE-STEP',
            'target': 50000,
            'nr_workers': 4,
            'nr_predictors': 2,
            'predictor_output_names': ['q_argmax'],
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
        'learning_rate': 0.0001,
        'nr_epochs': 800,

        # Parameters for Q-learner.
        'batch_size': 64,
        'epoch_size': 4000,
    }
}


def make_network(env):
    is_train = env.phase is env.Phase.TRAIN

    with env.create_network() as net:
        h, w, c = get_input_shape()

        dpc = env.create_dpcontroller()
        with dpc.activate():
            def inputs():
                state = O.placeholder('state', shape=(None, h, w, c))
                next_state = O.placeholder('next_state', shape=(None, h, w, c))
                return [state, next_state]

            @O.auto_reuse
            def phi(x):
                _ = x / 255.0

                # Nature structure
                with O.argscope(O.conv2d, nonlin=O.relu):
                    _ = O.conv2d('conv1', _, 32, 8, stride=4)
                    _ = O.conv2d('conv2', _, 64, 4, stride=2)
                    _ = O.conv2d('conv3', _, 64, 3, stride=1)
                return _

            def forward(state, next_state):
                dpc.add_output(phi(state), name='feature')
                dpc.add_output(phi(next_state), name='next_feature')

            dpc.set_input_maker(inputs).set_forward_func(forward)

        @O.auto_reuse
        def phi_fc(feature):
            _ = feature
            _ = O.fc('fc0', _, 512, nonlin=functools.partial(O.leaky_relu, alpha=0.01))
            q_pred = O.fc('fcq', _, get_player_nr_actions())
            q_max = q_pred.max(axis=1)
            q_argmax = q_pred.argmax(axis=1)
            return q_pred, q_max, q_argmax

        _ = dpc.outputs['feature']
        q_pred, q_max, q_argmax = phi_fc(_)

        _ = dpc.outputs['next_feature']
        next_q_pred, next_q_max, _ = phi_fc(_)

        net.add_output(q_pred, name='q_pred')
        net.add_output(q_max, name='q_max')
        net.add_output(q_argmax, name='q_argmax')

        if is_train:
            reward = O.placeholder('reward', shape=(None, ), dtype='float32')
            action = O.placeholder('action', shape=(None, ), dtype='int64')
            is_over = O.placeholder('is_over', shape=(None, ), dtype='bool')

            assert get_env('dqn.nr_td_steps') == 1
            this_q_pred = (q_pred * O.one_hot(action, get_player_nr_actions())).sum(axis=1)
            this_q_label = reward + get_env('dqn.gamma') * (1 - is_over.astype('float32')) * O.zero_grad(next_q_max)

            summary.scalar('this_q_pred', this_q_pred.mean())
            summary.scalar('this_q_label', this_q_label.mean())
            summary.scalar('reward', reward.mean())
            summary.scalar('is_over', is_over.astype('float32').mean())

            q_loss = O.raw_smooth_l1_loss('raw_q_loss', this_q_pred, this_q_label).mean(name='q_loss')
            net.set_loss(q_loss)


def make_player(is_train=True, dump_dir=None):
    def resize_state(s):
        return image.grayscale(image.resize(s, get_env('dqn.input_shape'), interpolation='NEAREST'))

    p = rl.GymAtariRLEnviron(get_env('dqn.env_name'), live_lost_as_eoe=is_train, dump_dir=dump_dir)
    p = rl.RepeatActionProxyRLEnviron(p, get_env('dqn.frame_skip'))
    p = rl.MapStateProxyRLEnviron(p, resize_state)
    p = rl.HistoryFrameProxyRLEnviron(p, get_env('dqn.nr_history_frames'))

    p = rl.LimitLengthProxyRLEnviron(p, get_env('dqn.max_nr_steps'))
    if not is_train:
        p = rl.GymPreventStuckProxyRLEnviron(p, get_env('dqn.inference.max_antistuck_repeat'), 1)
    return p


def make_optimizer(env):
    lr = optimizer.base.make_optimizer_variable('learning_rate', get_env('trainer.learning_rate'))

    wrapper = optimizer.OptimizerWrapper()
    wrapper.set_base_optimizer(optimizer.base.RMSPropOptimizer(lr, epsilon=1e-3))
    wrapper.append_grad_modifier(optimizer.grad_modifier.LearningRateMultiplier([
        ('*/b', 2.0),
    ]))
    wrapper.append_grad_modifier(optimizer.grad_modifier.WeightDecay([
        ('*/W', 5e-4),
    ]))
    env.set_optimizer(wrapper)


def make_dataflow_train(env):
    rng = random.gen_rng()

    def _outputs2action(outputs):
        epsilon = env.runtime['exp_epsilon']
        return outputs['q_argmax'] if rng.rand() > epsilon else rng.choice(get_player_nr_actions())

    collector = rl.train.SynchronizedExperienceCollector(
        env, make_player, _outputs2action,
        nr_workers=get_env('dqn.collector.nr_workers'), nr_predictors=get_env('dqn.collector.nr_workers'),
        predictor_output_names=get_env('dqn.collector.predictor_output_names'),
        mode=get_env('dqn.collector.mode')
    )

    return rl.train.QLearningDataFlow(
        collector, target=get_env('dqn.collector.target'), maxsize=get_env('dqn.expreplay.maxsize'),
        batch_size=get_env('trainer.batch_size'), epoch_size=get_env('trainer.epoch_size'),
        gamma=get_env('dqn.gamma'), nr_td_steps=get_env('dqn.nr_td_steps'),
        reward_cb=lambda r: np.clip(r, -1, 1))


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
    h, w, c = input_shape[0], input_shape[1], 1 * nr_history_frames
    return h, w, c


def main_inference_play_multithread(trainer):
    def runner():
        func = trainer.env.make_func()
        func.compile(trainer.env.network.outputs['q_argmax'])
        player = make_player()
        score = player.evaluate_one_episode(lambda state: func(state=state[np.newaxis])[0])

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
        'train/exp_epsilon': 'async_scalar'
    })
    summary.enable_echo_summary_scalar(trainer, summary_spec={
        'inference/score': ['avg', 'max']
    })

    from tartist.plugins.trainer_enhancer import progress
    progress.enable_epoch_progress(trainer)

    from tartist.plugins.trainer_enhancer import snapshot
    snapshot.enable_snapshot_saver(trainer, save_interval=2)

    def set_exp_epsilon(trainer_or_env, value):
        r = trainer_or_env.runtime
        if r.get('exp_epsilon', None) != value:
            logger.critical('Setting exploration epsilon to {}'.format(value))
            r['exp_epsilon'] = value

    schedule = [(0, 0.1), (10, 0.1), (250, 0.01), (1e9, 0.01)]
    def schedule_exp_epsilon(trainer):
        # `trainer.runtime` is synchronous with `trainer.env.runtime`.
        last_v = None
        for e, v in schedule:
            if trainer.epoch < e:
                set_exp_epsilon(trainer, last_v)
                break
            last_v = v

    def on_epoch_after(trainer):
        if trainer.epoch > 0 and trainer.epoch % 2 == 0:
            main_inference_play_multithread(trainer)

        # Summarize the exp epsilon.
        mgr = trainer.runtime.get('summary_histories', None)
        if mgr is not None:
            mgr.put_async_scalar('train/exp_epsilon', trainer.runtime['exp_epsilon'])

    # This one should run before monitor.
    trainer.register_event('epoch:before', schedule_exp_epsilon, priority=5)
    trainer.register_event('epoch:after', on_epoch_after, priority=5)

    trainer.train()


def main_demo(env, func):
    func.compile(env.network.outputs['q_argmax'])

    dump_dir = get_env('dir.demo', os.path.join(get_env('dir.root'), 'demo'))
    logger.info('Demo dump dir: {}'.format(dump_dir))
    player = make_player(dump_dir=dump_dir)
    repeat_time = get_env('dqn.demo.nr_plays', 1)

    for i in range(repeat_time):
        player.play_one_episode(func=lambda state: func(state=state[np.newaxis])[0])
        logger.info('#{} play score={}'.format(i, player.stats['score'][-1]))
