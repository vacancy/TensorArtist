# -*- coding:utf8 -*-
# File   : remote.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/10/17
# 
# This file is part of TensorArtist

from .base import SimpleDataFlowBase
from ..rflow import InputPipe, control

__all__ = ['RemoteFlow']


class RemoteFlow(SimpleDataFlowBase):
    def __init__(self, pipe_name, bufsize=100):
        self._pipe = InputPipe(pipe_name, bufsize=bufsize)

    def _gen(self):
        with control([self._pipe]):
            while True:
                yield self._pipe.get()

