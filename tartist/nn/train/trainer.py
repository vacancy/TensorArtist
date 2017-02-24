# -*- coding:utf8 -*-
# File   : trainer.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 1/31/17
# 
# This file is part of TensorArtist

from .env import TrainerEnv

__all__ = ['Trainer']


class Trainer(object):
    def __init__(self, env=None):
        self._env = env or TrainerEnv()
