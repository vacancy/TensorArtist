# -*- coding:utf8 -*-
# File   : data_provider.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/30/16
#
# This file is part of TensorArtist

from tartist.core import get_env
from tartist.data import flow
from tartist.data.datasets.cifar import load_cifar

import numpy as np

_cifar = []
SIZE = 32

def ensure_load(cifar_num_classes):
    global _cifar

    if len(_cifar) == 0:
        for xy in load_cifar(get_env('dir.data'), cifar_num_classes):
            _cifar.append(dict(img=xy[0].astype('float32').reshape(-1, SIZE, SIZE, 3), label=xy[1]))

def make_dataflow_train(env):
    num_classes = get_env('dataset.num_classes')
    ensure_load(num_classes)
    batch_size = get_env('trainer.batch_size')

    #from IPython import embed; embed()

    df = _cifar[0]
    df = flow.DOARandomSampleDataFlow(df)
    df = flow.BatchDataFlow(df, batch_size, sample_dict={
        'img': np.empty(shape=(batch_size, SIZE, SIZE, 3), dtype='float32'),
        'label': np.empty(shape=(batch_size, ), dtype='int32')
    })

    return df

