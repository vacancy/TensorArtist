# -*- coding:utf8 -*-
# File   : data_provider.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/30/16
# 
# This file is part of TensorArtist

from tartist.core import get_env
from tartist.data import flow 
from tartist.data.datasets.mnist import load_mnist

import numpy as np

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
    df = flow.BatchDataFlow(df, batch_size, sample_dict={
        'img': np.empty(shape=(batch_size, 28, 28, 1), dtype='float32'),
        'label': np.empty(shape=(batch_size, ), dtype='int32')
    })

    return df


def make_dataflow_inference(env):
    ensure_load()
    batch_size = get_env('inference.batch_size')
    epoch_size = get_env('inference.epoch_size')

    df = _mnist[1]  # use validation set actually
    df = flow.DictOfArrayDataFlow(df)
    df = flow.tools.cycle(df)
    df = flow.BatchDataFlow(df, batch_size, sample_dict={
        'img': np.empty(shape=(batch_size, 28, 28, 1), dtype='float32'),
        'label': np.empty(shape=(batch_size, ), dtype='int32')
    })
    df = flow.EpochDataFlow(df, epoch_size)

    return df

