# -*- coding:utf8 -*-
# File   : common_hp_a3c.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 06/07/2017
# 
# This file is part of TensorArtist.

import threading

from tartist.core import get_env


def main_inference_play_multithread(trainer, make_player, inpkey='state', policykey='policy'):
    def runner():
        func = trainer.env.make_func()
        func.compile(trainer.env.network.outputs[policykey])
        player = make_player(is_train=False)

        if isinstance(player.action_space, rl.DiscreteActionSpace):
            is_continuous = False
        elif isinstance(player.action_space, rl.ContinuousActionSpace):
            is_continuous = True
        else:
            raise AttributeError('Unknown action space: {}'.format(player.action_space))

        def get_action(inp, func=func):
            action = func(**{inpkey: inp[np.newaxis]})[0]
            if not is_continuous:
                return action.argmax()
            return action

        player.play_one_episode(get_action)
        score = player.stats['score'][-1]

        mgr = trainer.runtime.get('summary_histories', None)
        if mgr is not None:
            mgr.put_async_scalar('async/inference/score', score)

    nr_players = get_env('a3c.inference.nr_plays')
    pool = [threading.Thread(target=runner) for _ in range(nr_players)]
    for p in pool:
        p.start()
    for p in pool:
        p.join()
