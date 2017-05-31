# -*- coding:utf8 -*-
# File   : common_a3c.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/21/17
# 
# This file is part of TensorArtist

import collections
import threading

import numpy as np

from tartist.core import get_env

PlayerHistory = collections.namedtuple('PlayerHistory', ('state', 'action', 'value', 'reward'))


def on_data_func(env, identifier, inp_data):
    router, task_queue = env.player_master.router, env.player_master.queue
    data_queue = env.data_queue
    player_history = env.players_history[identifier]

    state, reward, is_over = inp_data

    def parse_history(history, is_over):
        num = len(history)
        if is_over:
            r = 0
            env.players_history[identifier] = []
        elif num == get_env('a3c.acc_step') + 1:
            history, last = history[:-1], history[-1]
            r = last.value
            env.players_history[identifier] = [last]
        else:
            return

        gamma = get_env('a3c.gamma')
        for i in history[::-1]:
            r = np.clip(i.reward, -1, 1) + gamma * r
            data_queue.put({'state': i.state, 'action': i.action, 'future_reward': r})

    def callback(action, predict_value):
        player_history.append(PlayerHistory(state, action, predict_value, None))
        router.send(identifier, action)

    if len(player_history) > 0:
        last = player_history[-1]
        player_history[-1] = PlayerHistory(last[0], last[1], last[2], reward)
        parse_history(player_history, is_over)

    task_queue.put((identifier, inp_data, callback))


def inference_on_data_func(env, identifier, inp_data):
    router, task_queue = env.inference_player_master.router, env.inference_player_master.queue

    def callback(action, _):
        router.send(identifier, action)

    task_queue.put((identifier, inp_data, callback))


def on_stat_func(env, identifier, inp_data):
    if env.owner_trainer is not None:
        mgr = env.owner_trainer.runtime.get('summary_histories', None)
        if mgr is not None:
            for k, v in inp_data.items():
                mgr.put_async_scalar(k, v)


inference_on_stat_func = on_stat_func


def main_inference_play(trainer, epoch):
    nr_players = get_env('a3c.inference.nr_players')
    name = 'a3c-inference-player-epoch-{}'.format(epoch)
    trainer.env.inference_player_master.start(nr_players, name=name, daemon=False)


def main_inference_play_multithread(trainer, make_player):
    def runner():
        func = trainer.env.make_func()
        func.compile(trainer.env.network.outputs)
        player = make_player(is_train=False)

        def get_action(inp, func=func):
            action = func(state=inp[np.newaxis])['policy'][0].argmax()
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

