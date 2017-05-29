# -*- coding:utf8 -*-
# File   : data_provider_vae_mnist.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/30/16
# 
# This file is part of TensorArtist

import numpy as np

from tartist import image
from tartist.core import get_env
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

    return df


def make_dataflow_inference(env):
    ensure_load()
    batch_size = get_env('inference.batch_size')
    epoch_size = get_env('inference.epoch_size')

    df = _mnist[1]  # use validation set actually
    df = flow.DictOfArrayDataFlow(df)
    df = flow.tools.cycle(df)
    df = flow.BatchDataFlow(df, batch_size,
                            sample_dict={'img': np.empty(shape=(batch_size, 28, 28, 1), dtype='float32'), })
    df = flow.EpochDataFlow(df, epoch_size)

    return df


def make_dataflow_demo(env):
    reconstruct = get_env('demo.is_reconstruct', False)
    if reconstruct:
        ensure_load()

        # return feed_dict, extra_info
        def split_data(img, label):
            return dict(img=img[np.newaxis].astype('float32'))

        df = _mnist[1]  # use validation set actually
        df = flow.DictOfArrayDataFlow(df)
        df = flow.tools.cycle(df)
        df = flow.tools.ssmap(split_data, df)
    else:
        df = flow.EmptyDictDataFlow()

    return df


def demo(feed_dict, result, extra_info):
    mode = get_env('demo.mode', 'vae')
    assert mode in ('vae', 'draw')

    if mode == 'vae':
        demo_vae(feed_dict, result, extra_info)
    elif mode == 'draw':
        demo_draw(feed_dict, result, extra_info)


def demo_vae(feed_dict, result, extra_info):
    reconstruct = get_env('demo.is_reconstruct', False)
    if reconstruct:
        img = feed_dict['img'][0, :, :, 0]
        omg = result['output'][0, :, :, 0]
        img = np.hstack((img, omg))
    else:
        img = result['output'][0, :, :, 0]

    img = np.repeat(img[:, :, np.newaxis], 3, axis=2) * 255
    img = img.astype('uint8')
    img = image.resize_minmax(img, 256)

    image.imshow('demo', img)


def demo_draw(feed_dict, result, extra_info):
    reconstruct = get_env('demo.is_reconstruct', False)
    grid_desc = get_env('demo.draw.grid_desc')

    all_outputs = []
    for i in range(1000):
        name = 'canvas_step{}'.format(i)
        if name in result:
            all_outputs.append(result[name][0, :, :, 0])

    final = image.image_grid(all_outputs, grid_desc)
    final = (final * 255).astype('uint8')

    if reconstruct:
        img = feed_dict['img'][0, :, :, 0]
        h = final.shape[0]
        w = int(img.shape[1] * h / img.shape[0])
        img = (img * 255).astype('uint8')
        img = image.resize(img, (h, w))
        final = np.hstack((img, final))

    final = image.resize_minmax(final, 480, 720)

    image.imshow('demo', final)
