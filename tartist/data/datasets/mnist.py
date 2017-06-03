# -*- coding:utf8 -*-
# File   : mnist.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/26/17
# 
# This file is part of TensorArtist.

from ...core import io

import os
import gzip
import pickle

__all__ = ['load_mnist']


def load_mnist(data_dir, 
        data_file='mnist.pkl.gz',
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'):

    dataset = os.path.join(data_dir, data_file)

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        io.download(origin, data_dir, data_file)

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    return train_set, valid_set, test_set
