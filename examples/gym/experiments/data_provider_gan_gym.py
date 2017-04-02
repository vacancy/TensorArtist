# -*- coding:utf8 -*-
# File   : data_provider_gan_gym.py
# Author : Jiayuan Mao
#          Honghua Dong
# Email  : maojiayuan@gmail.com
#          dhh19951@gmail.com
# Date   : 3/29/17
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
    def __init__(self, player, output_next=False):
        self.player = player
        self.player.restart()
        self.output_next = output_next

    def _gen(self):
        state = None
        counter = 0
        while True:
            action = random.choice(get_player_nr_actions())
            self.player.action(action)
            next_state = self.player._get_current_state()
            if counter < get_env('gym.frame_history'):
                counter += 1
            else:
                if self.output_next:
                    yield {'state': state, 'next_state': next_state[:, :, -3:]}
                else:
                    yield {'state': state}
            state = next_state


def make_dataflow_train(env):
    batch_size = get_env('trainer.batch_size')
    h, w, c = get_input_shape()

    df_g = MyDataFlow(make_player(), output_next=False)
    df_g = flow.BatchDataFlow(df_g, batch_size, sample_dict={
        'state': np.empty(shape=(batch_size, h, w, c), dtype='float32')
    })

    df_d = MyDataFlow(make_player(), output_next=True)
    df_d = flow.BatchDataFlow(df_d, batch_size, sample_dict={
        'state': np.empty(shape=(batch_size, h, w, c), dtype='float32'),
        'next_state': np.empty(shape=(batch_size, h, w, 3), dtype='float32'),
    })
    df = train.gan.GANDataFlow(df_g, df_d, get_env('trainer.nr_g_per_iter', 1), get_env('trainer.nr_d_per_iter', 1))

    return df


# does not support inference during training
def make_dataflow_inference(env):
    df = flow.tools.cycle([{'d': [], 'g': []}])
    return df


def make_dataflow_demo(env):
    df = MyDataFlow(make_player(), output_next=False)
    df = flow.BatchDataFlow(df, 1, sample_dict={
        'state': np.empty(shape=(1, h, w, c), dtype='float32')
    })
    return df


def demo(feed_dict, result, extra_info):
    n = get_env('gym.frame_history')
    states = feed_dict['state'][0]
    assert(len(states.shape()) == 3)
    states = np.split(states, n, axis=2)
    pred = result['output'][0]
    img = np.hstack((states + pred))
    print(img.shape())

    img = img * 255
    img = img.astype('uint8')
    img = image.resize_minmax(img, 256)

    image.imshow('demo', img)
