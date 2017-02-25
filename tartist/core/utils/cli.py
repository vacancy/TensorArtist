# -*- coding:utf8 -*-
# File   : cli.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/26/17
# 
# This file is part of TensorArtist

from .imp import load_module_filename
from ..environ import load_env

import os
import sys


def parse_devices(devs):
    all_gpus = []
    def rename(d):
        d = d.lower()
        if d == 'cpu':
            return '/cpu:0'
        if d.startswith('gpu'):
            d = d[3:]
        d = str(int(d))
        all_gpus.append(d)
        return '/gpu:' + d
    devs = tuple(map(rename, devs))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(all_gpus)
    return devs


def load_desc(desc_filename):
    module = load_module_filename(desc_filename)
    load_env(module)
    return module

