# -*- coding:utf8 -*-
# File   : env.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/29/16
# 
# This file is part of TensorArtist

from ...core.utils.meta import assert_notnone
from ..graph.env import Env
from ..graph.node import as_tftensor

__all__ = ['TrainerEnv']


class TrainerEnv(Env):
    _optimizer = None
    _fn_train_step = None

    @property
    def optimizer(self):
        assert_notnone(self._optimizer, name='trainer_env.optimzier')
        return self._optimizer

    def set_optimizer(self, opt):
        self._optimizer = opt

    def make_optimizable_func(self):
        func = self.make_func()
        loss = as_tftensor(self.network.loss)
        func.add_extra_outputs(self.optimizer.minimize(loss))
        return func
