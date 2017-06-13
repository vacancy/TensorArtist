# -*- coding:utf8 -*-
# File   : cifar.py
# Author : Jiayuan Mao
#          Honghua Dong
# Email  : maojiayuan@gmail.com
#          dhh19951@gmail.com
# Date   : 2/27/17
#
# This file is part of TensorArtist.

from ...core import io

import os

import pickle
import tarfile

import numpy as np

__all__ = ['load_cifar']

cifar_web_address = 'http://www.cs.toronto.edu/~kriz/'

def _read_cifar(filenames, cls):
    image = []
    label = []
    for fname in filenames:
        with open(fname, 'rb') as f:
            raw_dict = pickle.load(f, encoding='latin1')
        raw_data = raw_dict['data']
        label.extend(raw_dict['labels' if cls == 10 else 'fine_labels'])
        for x in raw_data:
            x = x.reshape(3, 32, 32)
            x = np.transpose(x, [1, 2, 0])
            image.append(x)
    return np.array(image), np.array(label)


def load_cifar(data_dir, cls=10):
    assert cls in [10, 100]

    data_file = 'cifar-{}-python.tar.gz'.format(cls)
    origin = cifar_web_address + data_file
    dataset = os.path.join(data_dir, data_file)
    if cls == 10:
        folder_name = 'cifar-10-batches-py'
        filenames = ['data_batch_{}'.format(i) for i in range(1, 6)]
        filenames.append('test_batch')
    else:
        folder_name = 'cifar-100-python'
        filenames = ['train', 'test']

    if not os.path.isdir(os.path.join(data_dir, folder_name)):
        if not os.path.isfile(dataset):
            io.download(origin, data_dir, data_file)
        tarfile.open(dataset, 'r:gz').extractall(data_dir)

    filenames = list(map(lambda x : os.path.join(data_dir, folder_name, x), filenames))

    train_set = _read_cifar(filenames[:-1], cls)
    test_set = _read_cifar([filenames[-1]], cls)

    return train_set, test_set
