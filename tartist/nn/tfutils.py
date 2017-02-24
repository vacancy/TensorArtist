# -*- coding:utf8 -*-
# File   : tfutils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 1/31/17
# 
# This file is part of TensorArtist


def clean_name(tensor, suffix=':0'):
    name = tensor.name
    if name.endswith(suffix):
        name = name[:-len(suffix)]
    return name
