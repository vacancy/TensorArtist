# -*- coding:utf8 -*-
# File   : env.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/29/16
# 
# This file is part of TensorArtist

from ...core.utils.meta import notnone_property
from ..graph.env import Env
from ..graph.node import as_tftensor

__all__ = ['TrainerEnv']


class TrainerEnv(Env):
    _optimizer = None
    _data_provider = None
    _fn_train_step = None

    @notnone_property
    def optimizer(self):
        return self._optimizer

    def set_optimizer(self, opt):
        self._optimizer = opt
        return self

    def make_optimizable_func(self, loss=None):
        loss = loss or self.network.loss
        loss = as_tftensor(loss)

        func = self.make_func()
        func.add_extra_op(self.optimizer.minimize(loss))
        return func
