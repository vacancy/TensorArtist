# -*- coding:utf8 -*-
# File   : data_provider_discogan.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 4/2/17
# 
# This file is part of TensorArtist.

import os.path as osp

import numpy as np

from tartist import image
from tartist.app import gan
from tartist.core import get_env
from tartist.data import flow, kvstore


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
            res = dict(img_a=self._crop_and_resize(next(ita)), img_b=self._crop_and_resize(next(itb)))
            yield res


def _make_dataflow(batch_size=1, use_prefetch=False):
    img_shape = get_env('dataset.img_shape', (64, 64))

    data_dir = get_env('dir.data')
    db_a = osp.join(data_dir, get_env('dataset.db_a'))
    db_b = osp.join(data_dir, get_env('dataset.db_b'))

    assert osp.exists(db_a) and osp.exists(db_b), ('Unknown database: {} and {}. If you haven\'t downloaded them,'
        'please run scripts in TensorArtist/scripts/dataset-tools/pix2pix and put the generated dataset lmdb'
        'in the corresponding position.'.format(db_a, db_b))

    dfs = []
    dfa = flow.KVStoreRandomSampleDataFlow(lambda: kvstore.LMDBKVStore(db_a))
    dfb = flow.KVStoreRandomSampleDataFlow(lambda: kvstore.LMDBKVStore(db_b))
    df = DiscoGANSplitDataFlow(dfa, dfb)
    df = flow.BatchDataFlow(df, batch_size, sample_dict={
        'img_a': np.empty(shape=(batch_size, img_shape[0], img_shape[1], 3), dtype='float32'),
        'img_b': np.empty(shape=(batch_size, img_shape[0], img_shape[1], 3), dtype='float32'), })
    if use_prefetch:
        df = flow.MPPrefetchDataFlow(df, nr_workers=2)
    return df

    df = gan.GANDataFlow(dfs[0], dfs[1], get_env('trainer.nr_g_per_iter', 1), get_env('trainer.nr_d_per_iter', 1))


def make_dataflow_train(env):
    batch_size = get_env('trainer.batch_size')
    dfs = [_make_dataflow(batch_size, use_prefetch=True) for i in range(2)]

    df = gan.GANDataFlow(dfs[0], dfs[1], get_env('trainer.nr_g_per_iter', 1), get_env('trainer.nr_d_per_iter', 1))

    return df


def make_dataflow_demo(env):
    return _make_dataflow(1)


def main_demo(env, func):
    df = iter(make_dataflow_demo(env))
    nr_samples = get_env('demo.nr_samples', 40 * 8)
    grid_desc = get_env('demo.grid_desc', ('20v', '16h'))
    
    while True:
        all_imgs_ab = []
        all_imgs_ba = []
        for i in range(nr_samples):
            feed_dict = next(df)
            results = func(**feed_dict)
            img_a, img_b = feed_dict['img_a'][0], feed_dict['img_b'][0]
            img_ab, img_ba = results['img_ab'][0] * 255, results['img_ba'][0] * 255
            img_aba, img_bab = results['img_aba'][0] * 255, results['img_bab'][0] * 255

            all_imgs_ab.append(np.hstack([img_a, img_ab]).astype('uint8'))
            all_imgs_ba.append(np.hstack([img_b, img_ba]).astype('uint8'))

        all_imgs_ab = image.image_grid(all_imgs_ab, grid_desc)
        all_imgs_ba = image.image_grid(all_imgs_ba, grid_desc)
        sep = np.ones((all_imgs_ab.shape[0], 64, 3), dtype='uint8') * 255
        all_imgs = np.hstack([all_imgs_ab, sep, all_imgs_ba])
        image.imwrite('discogan.png', all_imgs)
        image.imshow('AtoB; BtoA', all_imgs)

