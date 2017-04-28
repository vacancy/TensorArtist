# -*- coding:utf8 -*-
# File   : embed-dataflow.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 4/27/17
# 
# This file is part of TensorArtist

from tartist import image
from tartist.core import get_env, get_logger, set_env
from tartist.core.utils.cli import load_desc, parse_devices
from tartist.nn import Env, train

import os
import time
import argparse
import tensorflow as tf
import numpy as np

logger = get_logger(__file__)

parser = argparse.ArgumentParser()
parser.add_argument(dest='desc', help='The description file module')
args = parser.parse_args()

# begin: all print tools
from pprint import pprint


def imshow(img, resize=(600, 800), title='imshow'):
    img = image.resize_minmax(img, *resize, interpolation='NEAREST')
    image.imshow(title, img)


def batch_show(batch, nr_show=16, grid_desc=('4h', '4v'), resize=(600, 800), title='batch_show'):
    batch = batch[:nr_show]
    if len(batch) < 16:
        batch = np.concatenate([batch, 
                np.zeros([16 - len(batch), batch.shape[1], batch.shape[2], batch.shape[3]], dtype=batch.dtype)
        ], axis=0)

    img = image.image_grid(batch, grid_desc)
    img = image.resize_minmax(img, *resize, interpolation='NEAREST')
    image.imshow(title, img)


def _indent_print(msg, indent, prefix=None, end='\n'):
    print(*['  '] * indent, end='')
    if prefix is not None:
        print(prefix, end='')
    print(msg, end=end)


def struct_show(data, key=None, indent=0):
    t = type(data)

    if t is tuple:
        _indent_print('tuple[', indent, prefix=key)
        for v in data:
            struct_show(v, indent=indent + 1)
        _indent_print(']', indent)
    elif t is list:
        _indent_print('list[', indent, prefix=key)
        for v in data:
            struct_show(v, indent=indent + 1)
        _indent_print(']', indent)
    elif t is dict:
        _indent_print('dict{', indent, prefix=key)
        for k, v in data.items():
            struct_show(v, indent=indent + 1, key='{}: '.format(k))
        _indent_print('}', indent)
    elif t is np.ndarray:
        _indent_print('ndarray{}, dtype={}'.format(data.shape, data.dtype), indent, prefix=key)
    else:
        _indent_print(data, indent, prefix=key)


def main():
    desc = load_desc(args.desc)

    env = Env(Env.Phase.TRAIN, master_dev='/cpu:0')
    df = desc.make_dataflow_train(env)
    for data in df:
        struct_show(data)
        print('Accessible variables: env, df, data.')
        print('Try: pprint(v), imshow(img), batch_show(batch), struct_show(object).')
        from IPython import embed; embed()


if __name__ == '__main__':
    main()

