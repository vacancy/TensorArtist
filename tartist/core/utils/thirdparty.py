# -*- coding:utf8 -*-
# File   : thirdparty.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/23/17
# 
# This file is part of TensorArtist.

__all__ = ['get_tqdm_defaults']


__tqdm_defaults = {'dynamic_ncols': True, 'ascii': True}

def get_tqdm_defaults():
    return __tqdm_defaults
