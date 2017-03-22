# -*- coding:utf8 -*-
# File   : common_a3c.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/21/17
# 
# This file is part of TensorArtist

import time
import queue
import threading
import collections
import numpy as np
import tensorflow as tf

from tartist.core import get_env, get_logger, EnvBox
from tartist.core.utils.cache import cached_result
from tartist.core.utils.naming import get_dump_directory, get_data_directory
from tartist.data import flow
from tartist.data.rflow import QueryReqPipe
from tartist.nn import opr as O, optimizer, summary
from tartist import rl, random, image

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
        router.send(identifier, action)
        player_history.append(PlayerHistory(state, action, predict_value, None))

    task_queue.put((identifier, inp_data, callback))

    if len(player_history) > 0:
        last = player_history[-1]
        player_history[-1] = PlayerHistory(last[0], last[1], last[2], reward)
        parse_history(player_history, is_over)


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

