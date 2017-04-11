# -*- coding:utf8 -*-
# File   : data_provider_discogan.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 4/2/17
# 
# This file is part of TensorArtist

from tartist import image
from tartist.core import get_env
from tartist.core.utils.thirdparty import get_tqdm_defaults
from tartist.data import flow, kvstore
from tartist.nn import train

import numpy as np
import tqdm
import os.path as osp


class DiscoGANSplitDataFlow(flow.SimpleDataFlowBase):
    def __init__(self, kva, kvb):
        super().__init__()
        self._kva = kva
        self._kvb = kvb
        self._img_shape = get_env('dataset.img_shape', (64, 64))

    def _crop_and_resize(self, img):
        img = image.imdecode(img)
        img = image.resize(img, self._img_shape)
        return img

    def _gen(self):
        ita = iter(self._kva)
        itb = iter(self._kvb)
        while True:
            res = dict(
                img_a=self._crop_and_resize(next(ita)),
                img_b=self._crop_and_resize(next(itb))
            )
            yield res


def _make_dataflow(batch_size=1, use_prefetch=False):
    img_shape = get_env('dataset.img_shape', (64, 64))

    data_dir = get_env('dir.data')
    db_a = osp.join(data_dir, get_env('dataset.db_a'))
    db_b = osp.join(data_dir, get_env('dataset.db_b'))

    dfs = []
    dfa = flow.KVStoreRandomSampleDataFlow(lambda: kvstore.LMDBKVStore(db_a))
    dfb = flow.KVStoreRandomSampleDataFlow(lambda: kvstore.LMDBKVStore(db_b))
    df = DiscoGANSplitDataFlow(dfa, dfb)
    df = flow.BatchDataFlow(df, batch_size, sample_dict={
        'img_a': np.empty(shape=(batch_size, img_shape[0], img_shape[1], 3), dtype='float32'),
        'img_b': np.empty(shape=(batch_size, img_shape[0], img_shape[1], 3), dtype='float32'),
    })
    if use_prefetch:
        df = flow.MPPrefetchDataFlow(df, nr_workers=2)
    return df


def make_dataflow_train(env):
    batch_size = get_env('trainer.batch_size')
    dfs = [_make_dataflow(batch_size, use_prefetch=True) for i in range(2)]

    df = train.gan.GANDataFlow(dfs[0], dfs[1], 
            get_env('trainer.nr_g_per_iter', 1), get_env('trainer.nr_d_per_iter', 1))

    return df


def make_dataflow_demo(env):
    return _make_dataflow(1)


def demo(feed_dict, results, extra_info):
    img_a, img_b = feed_dict['img_a'][0], feed_dict['img_b'][0]
    img_ab, img_ba = results['img_ab'][0] * 255, results['img_ba'][0] * 255
    img_aba, img_bab = results['img_aba'][0] * 255, results['img_bab'][0] * 255

    img = np.vstack([
        np.hstack([img_a, img_ab, img_aba]), 
        np.hstack([img_b, img_ba, img_bab])
    ])
    img = img.astype('uint8')
    img = image.resize_minmax(img, 512, 2048)

    image.imshow('demo', img)

