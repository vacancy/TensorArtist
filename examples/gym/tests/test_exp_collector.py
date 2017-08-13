# -*- coding:utf8 -*-
# File   : test_exp_collector.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/08/2017
# 
# This file is part of TensorArtist.

from tartist.app import rl
from tartist.core.utils.cache import cached_result
from tartist.nn import Env, opr as O


def make_network(env):
    with env.create_network() as net:
        state = O.placeholder('state', shape=(None, ) + get_input_shape())
        logits = O.fc('fc', state, get_action_shape())
        net.add_output(logits, name='policy')


def make_player(dump_dir=None):
    p = rl.GymRLEnviron('CartPole-v0', dump_dir=dump_dir)
    p = rl.LimitLengthProxyRLEnviron(p, 200)
    return p


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


def make_experience_collector(env):
    collector = rl.train.SynchronizedExperienceCollector(
        owner_env=env, make_player=make_player, output2action=lambda x: int(x['policy'][0] > 0),
        nr_workers=8, nr_predictors=1
    )
    return collector


def main_test():
    env = Env(master_dev='/cpu:0')
    with env.as_default():
        make_network(env)
        env.initialize_all_variables()

    collector = make_experience_collector(env)

    collector.initialize()
    results = collector.run(target=200)

    from tartist import cao
    cao.embed()

if __name__ == '__main__':
    main_test()
