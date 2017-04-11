# -*- coding:utf8 -*-
# File   : data_provider_discogan.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 4/2/17
# 
# This file is part of TensorArtist

import os.path as osp

import numpy as np
import tqdm

from tartist import image
from tartist.core import get_env
from tartist.core.utils.thirdparty import get_tqdm_defaults
from tartist.data import flow, kvstore


class DiscoGANSplitDataFlow(flow.SimpleDataFlowBase):
    def __init__(self, kv):
        super().__init__()
        self._kv = kv

    def _crop_and_resize(self, img, part):
        if part == 0:
            img = img[:, :256]
        else:
            img = img[:, 256:]

        img = image.resize_minmax(img, 64)

    def _gen(self):
        it = iter(self._kv)
        while True:
            yield dict(
                img_a=self._crop_and_resize(next(it), 0),
                img_b=self._crop_and_resize(next(it), 1)
            )


def make_dataflow_train(env):
    batch_size = get_env('trainer.batch_size')

    assert get_env('dataset.name' == 'edges2shoes')
    data_dir = get_env('dir.data')

    kv = kvstore.LMDBKVStore(osp.join(data_dir, 'train_db'))

    dfs = []
    for i in range(2):
        df = flow.KVStoreRandomSampleDataFlow(kv)
        df = DiscoGANSplitDataFlow(df)
        df = flow.BatchDataFlow(df, batch_size, sample_dict={
            'img_a': np.empty(shape=(batch_size, 28, 28, 1), dtype='float32'),
            'img_b': np.empty(shape=(batch_size, 28, 28, 1), dtype='float32'),
        })
        dfs.append(df)

    df = tartist.app.gan.GANDataFlow(dfs[0], dfs[1],
                                     get_env('trainer.nr_g_per_iter', 1), get_env('trainer.nr_d_per_iter', 1))

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

