# -*- coding:utf8 -*-
# File   : data_provider_vae_mnist.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/30/16
# 
# This file is part of TensorArtist

import numpy as np
import tqdm

from tartist import image
from tartist.app import gan
from tartist.core import get_env
from tartist.core.utils.thirdparty import get_tqdm_defaults
from tartist.data import flow
from tartist.data.datasets.mnist import load_mnist

_mnist = []


def ensure_load():
    global _mnist

    if len(_mnist) == 0:
        for xy in load_mnist(get_env('dir.data')):
            _mnist.append(dict(img=xy[0].reshape(-1, 28, 28, 1), label=xy[1]))


def make_dataflow_train(env):
    ensure_load()
    batch_size = get_env('trainer.batch_size')

    df = _mnist[0]
    df = flow.DOARandomSampleDataFlow(df)
    df = flow.BatchDataFlow(df, batch_size,
                            sample_dict={'img': np.empty(shape=(batch_size, 28, 28, 1), dtype='float32'), })
    df = gan.GANDataFlow(None, df, get_env('trainer.nr_g_per_iter', 1), get_env('trainer.nr_d_per_iter', 1))

    return df


# does not support inference during training
def make_dataflow_inference(env):
    df = flow.tools.cycle([{'d': [], 'g': []}])
    return df


def make_dataflow_demo(env):
    df = flow.EmptyDictDataFlow()
    return df


def demo(feed_dict, result, extra_info):
    img = result['output'][0, :, :, 0]

    img = np.repeat(img[:, :, np.newaxis], 3, axis=2) * 255
    img = img.astype('uint8')
    img = image.resize_minmax(img, 256)

    image.imshow('demo', img)


def main_demo_infogan(env, func):
    net = env.network
    samples = net.zc_distrib.numerical_sample(net.zc_distrib_num_prior)
    df = {'zc': samples.reshape(samples.shape[0], 1, -1)}
    df = flow.DictOfArrayDataFlow(df)

    all_outputs = []
    for data in tqdm.tqdm(df, total=len(df), **get_tqdm_defaults()):
        res = func(**data)
        all_outputs.append(res['output'][0, :, :, 0])

    grid_desc = get_env('demo.infogan.grid_desc')
    final = image.image_grid(all_outputs, grid_desc)
    final = (final * 255).astype('uint8')
    image.imwrite('infogan.png', final)


def main_demo(env, func):
    mode = get_env('demo.mode')
    assert mode is not None

    if mode == 'infogan':
        main_demo_infogan(env, func)
    else:
        assert False, 'Unknown mode {}'.format(mode)
