# -*- coding:utf8 -*-
# File   : data_provider_atari.py
# Author : Jiayuan Mao
#          Honghua Dong
# Email  : maojiayuan@gmail.com
#          dhh19951@gmail.com
# Date   : 4/2/17
# 
# This file is part of TensorArtist

from tartist import image
from tartist.core import get_env
from tartist.core.utils.cache import cached_result
from tartist.core.utils.thirdparty import get_tqdm_defaults
from tartist.data import flow 
from tartist.data.datasets.mnist import load_mnist
from tartist.nn import train
from tartist import rl, random
import numpy as np
import tqdm
from collections import deque


def make_player():
    def resize_state(s):
        return image.resize(s, get_env('gym.input_shape'))

    p = rl.GymRLEnviron(get_env('gym.env_name'))
    p = rl.MapStateProxyRLEnviron(p, resize_state)
    p = rl.GymHistoryProxyRLEnviron(p, get_env('gym.frame_history'))
    p = rl.LimitLengthProxyRLEnviron(p, get_env('gym.limit_length'))
    p = rl.AutoRestartProxyRLEnviron(p)
    return p


@cached_result
def get_player_nr_actions():
    p = make_player()
    n = p.action_space.nr_actions
    del p
    return n


@cached_result
def get_input_shape():
    input_shape = get_env('gym.input_shape')
    frame_history = get_env('gym.frame_history')
    h, w, c = input_shape[0], input_shape[1], 3 * frame_history
    return h, w, c


class MyDataFlow(flow.SimpleDataFlowBase):
    def __init__(self, player, memory=10000):
        self._player = player
        self._player.restart()
        self._history = deque(maxlen=memory_size)
        #fill history
        self._player.action(1)
        for i in range(memory_size):
            self._history.append(self._get_data())

    def _get_data(self):
        state = self._player._get_current_state()
        action = random.choice(get_player_nr_actions())
        self._player.action(action)
        next_state = self._player._get_current_state()[:, :, -3:]
        return {'state': state, 'action': action, 'next_state': next_state}

    def _gen(self):
        while True:
            self._history.append(self._get_data())
            yield random.choice(self._history)


def make_dataflow_train(env):
    batch_size = get_env('trainer.batch_size')
    h, w, c = get_input_shape()

    df = MyDataFlow(make_player())
    df = flow.BatchDataFlow(df, batch_size, sample_dict={
        'state': np.empty(shape=(batch_size, h, w, c), dtype='float32'),
        'action': np.empty(shape=(batch_size, ), dtype='int64'),
        'next_state': np.empty(shape=(batch_size, h, w, 3), dtype='float32')
    })
    return df


def make_dataflow_inference(env):
    batch_size = get_env('inference.batch_size')
    epoch_size = get_env('inference.epoch_size')
    h, w, c = get_input_shape()

    df = MyDataFlow(make_player())
    df = flow.BatchDataFlow(df, batch_size, sample_dict={
        'state': np.empty(shape=(batch_size, h, w, c), dtype='float32'),
        'action': np.empty(shape=(batch_size, ), dtype='int64'),
        'next_state': np.empty(shape=(batch_size, h, w, 3), dtype='float32')
    })
    df = flow.EpochDataFlow(df, epoch_size)
    return df


def make_dataflow_demo(env):

    def split_data(state, action, next_state):
        return dict(state=state[np.newaxis].astype('float32'), action=[action]), dict(next_state=next_state)

    h, w, c = get_input_shape()
    df = MyDataFlow(make_player())
    df = flow.tools.ssmap(split_data, df)
    return df


def demo(feed_dict, result, extra_info):
    n = get_env('gym.frame_history')
    states = feed_dict['state'][0]
    next_state = extra_info['next_state']
    assert(len(states.shape) == 3)
    states = tuple(np.split(states, n, axis=2))
    pred = result['output'][0]
    pred = np.minimum(np.maximum(pred, 0), 1)
    pred = pred * 255.0
    img = np.hstack(states + (next_state, pred))
    img = img[:, : ,::-1]

    img = img.astype('uint8')
    img = image.resize_minmax(img, 256, 256 * (n + 2))

    image.imshow('demo', img)
